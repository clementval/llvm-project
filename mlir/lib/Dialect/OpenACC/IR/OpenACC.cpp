//===- OpenACC.cpp - OpenACC MLIR Operations ------------------------------===//
//
// Part of the MLIR Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// =============================================================================

#include "mlir/Dialect/OpenACC/OpenACC.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Function.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/Module.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/StandardTypes.h"
#include "mlir/IR/Value.h"

using namespace mlir;
using namespace acc;

//===----------------------------------------------------------------------===//
// OpenACCDialect
//===----------------------------------------------------------------------===//

OpenACCDialect::OpenACCDialect(MLIRContext *context)
    : Dialect(getDialectNamespace(), context) {
  addOperations<
#define GET_OP_LIST
#include "mlir/Dialect/OpenACC/OpenACC.cpp.inc"
      >();
  allowsUnknownOperations();
}

// Parses an op that has no inputs and no outputs.
static ParseResult parseNoIOOp(OpAsmParser &parser, OperationState &state) {
  if (parser.parseOptionalAttrDict(state.attributes))
    return failure();
  return success();
}

static void printNoIOOp(Operation *op, OpAsmPrinter &printer) {
  printer << op->getName();
  printer.printOptionalAttrDict(op->getAttrs());
}

template <typename StructureOp>
static ParseResult parseRegionOp(OpAsmParser &parser, OperationState &state,
                                 unsigned int nRegions = 1) {

  llvm::SmallVector<Region *, 2> regions;
  for (unsigned int i = 0; i < nRegions; ++i) {
    regions.push_back(state.addRegion());
  }

  for (auto &region : regions) {
    if (parser.parseRegion(*region, /*arguments=*/{}, /*argTypes=*/{})) {
      return failure();
    }
    StructureOp::ensureTerminator(*region, parser.getBuilder(), state.location);
  }

  if (succeeded(parser.parseOptionalKeyword("attributes"))) {
    if (parser.parseOptionalAttrDict(state.attributes))
      return failure();
  }

  return success();
}

static void printGangRedundantOp(GangRedundantOp &op, OpAsmPrinter &printer) {
  printer << GangRedundantOp::getOperationName();
  printer.printRegion(op.getBody(), false, false);
  printer.printOptionalAttrDictWithKeyword(op.getAttrs(), {});
}

template <typename StructureOp>
static ParseResult parseRegions(OpAsmParser &parser, OperationState &state,
                                unsigned int nRegions = 1) {

  llvm::SmallVector<Region *, 2> regions;
  for (unsigned int i = 0; i < nRegions; ++i) {
    regions.push_back(state.addRegion());
  }

  for (auto &region : regions) {
    if (parser.parseRegion(*region, /*arguments=*/{}, /*argTypes=*/{})) {
      return failure();
    }
    StructureOp::ensureTerminator(*region, parser.getBuilder(), state.location);
  }

  return success();
}

static ParseResult parseOptionalAttributes(OpAsmParser &parser,
                                           OperationState &state) {
  if (succeeded(parser.parseOptionalKeyword("attributes"))) {
    if (parser.parseOptionalAttrDict(state.attributes))
      return failure();
  }
  return success();
}

static ParseResult parseFormattedAttr(OpAsmParser &parser,
                                      OperationState &state, StringRef attrName,
                                      bool hasValue = true) {
  if (succeeded(parser.parseOptionalKeyword(attrName))) {
    Attribute attr;
    if (hasValue) {
      parser.parseLParen();
      parser.parseAttribute(attr, attrName, state.attributes);
      parser.parseRParen();
    } else {
      Builder &builder = parser.getBuilder();
      state.addAttribute(attrName, builder.getUnitAttr());
    }
  }
  return success();
}

template <typename OpTy>
static void printFormattedAttr(OpTy &op, OpAsmPrinter &printer,
                               StringRef attrName, bool hasValue = true) {
  if (op.getAttr(attrName) != nullptr) {
    printer << " " << attrName;
    if (hasValue) {
      printer << "(";
      printer.printAttribute(op.getAttr(attrName));
      printer << ")";
    }
  }
}

static ParseResult parseOperandList(OpAsmParser &parser, StringRef keyword,
                                    SmallVectorImpl<OpAsmParser::OperandType> &args,
                                    SmallVectorImpl<Type> &argTypes,
                                    OperationState &result) {
  if(failed(parser.parseOptionalKeyword(keyword)))
    return success();

  if (failed(parser.parseLParen()))
    return failure();

  // Exit if empty list
  if (succeeded(parser.parseOptionalRParen()))
    return success();

  do {
    OpAsmParser::OperandType arg;
    Type type;

    if (parser.parseRegionArgument(arg) || parser.parseColonType(type))
      return failure();

    args.push_back(arg);
    argTypes.push_back(type);
  } while (succeeded(parser.parseOptionalComma()));

  if (failed(parser.parseRParen()))
    return failure();

  for (auto operand_type : llvm::zip(args, argTypes)) {
    if (parser.resolveOperand(std::get<0>(operand_type),
                              std::get<1>(operand_type), result.operands))
      return failure();
  }
  return success();
}

static ParseResult parseOptionalOperand(OpAsmParser &parser, StringRef keyword,
                                        OpAsmParser::OperandType &operand,
                                        Type &type, bool &hasOptional,
                                        OperationState &result) {
  hasOptional = false;
  if(succeeded(parser.parseOptionalKeyword(keyword))) {
    hasOptional = true;
    if (parser.parseLParen() ||
        parser.parseOperand(operand) ||
        parser.resolveOperand(operand, type, result.operands) ||
        parser.parseRParen())
      return failure();
  }
  return success();
}

//===----------------------------------------------------------------------===//
// ParallelOp
//===----------------------------------------------------------------------===//

// Parse acc.parallel operation
// operation := `acc.parallel` `num_gangs(value)?` `num_workers(value)?`
//                             `vector_length(value)?` `private(value)?`
//                             `private(value-list)?` `firstprivate(value-list)`?
//                             region attr-dict?
static ParseResult parseParallelOp(OpAsmParser &parser,
                                   OperationState &result) {
  auto &builder = parser.getBuilder();
  SmallVector<OpAsmParser::OperandType, 8> privateOperands, 
      firstprivateOperands, createOperands, copyinOperands, copyoutOperands;
  SmallVector<Type, 8> operandTypes;
  OpAsmParser::OperandType numGangs, numWorkers, vectorLength, ifCond;
  bool hasNumGangs = false, hasNumWorkers = false, hasVectorLength = false,
      hasIfCond = false;

  Type indexType = builder.getIndexType();
  Type i1Type = builder.getI1Type();
  // num_gangs(value)?
  if(failed(parseOptionalOperand(parser, ParallelOp::getNumGangsKeyword(),
      numGangs, indexType, hasNumGangs, result)))
    return failure();

  // num_workers(value)?
  if(failed(parseOptionalOperand(parser, ParallelOp::getNumWorkersKeyword(),
      numWorkers, indexType, hasNumWorkers, result)))
    return failure();

  // vector_length(value)?
  if(failed(parseOptionalOperand(parser, ParallelOp::getVectorLengthKeyword(),
      vectorLength, indexType, hasVectorLength, result)))
    return failure();

  // private()?
  if (failed(parseOperandList(parser, ParallelOp::getPrivateKeyword(),
      privateOperands, operandTypes, result)))
    return failure();

  // firstprivate()?
  if (failed(parseOperandList(parser, ParallelOp::getFirstPrivateKeyword(),
      firstprivateOperands, operandTypes, result)))
    return failure();

  // create()?
  if (failed(parseOperandList(parser, ParallelOp::getCreateKeyword(),
      createOperands, operandTypes, result)))
    return failure();

  // copyin()?
  if (failed(parseOperandList(parser, ParallelOp::getCopyinKeyword(),
      copyinOperands, operandTypes, result)))
    return failure();

  // copyout()?
  if (failed(parseOperandList(parser, ParallelOp::getCopyoutKeyword(),
      copyoutOperands, operandTypes, result)))
    return failure();

  // if()?
  if(failed(parseOptionalOperand(parser, ParallelOp::getIfKeyword(),
      ifCond, i1Type, hasIfCond, result)))
    return failure();

  // Parallel op region
  if (failed(parseRegions<ParallelOp>(parser, result)))
    return failure();

  result.addAttribute(
      ParallelOp::getOperandSegmentSizeAttr(),
      builder.getI32VectorAttr({
          static_cast<int32_t>(hasNumGangs ? 1 : 0),
          static_cast<int32_t>(hasNumWorkers ? 1 : 0),
          static_cast<int32_t>(hasVectorLength ? 1 : 0),
          static_cast<int32_t>(hasIfCond ? 1 : 0),
          static_cast<int32_t>(privateOperands.size()),
          static_cast<int32_t>(firstprivateOperands.size()),
          static_cast<int32_t>(createOperands.size()),
          static_cast<int32_t>(copyinOperands.size()),
          static_cast<int32_t>(copyoutOperands.size())}));

  // Additional attributes
  if (failed(parseOptionalAttributes(parser, result)))
    return failure();

  return success();
}

static void printParallelOp(ParallelOp &op, OpAsmPrinter &printer) {
  printer << ParallelOp::getOperationName();

  if (auto numGangs = op.numGangs())
    printer << " " <<
        ParallelOp::getNumGangsKeyword() << "(" << numGangs << ")";

  if (auto numWorkers = op.numWorkers())
    printer << " " <<
        ParallelOp::getNumWorkersKeyword() << "(" << numWorkers << ")";

  // Gang private list
  if (op.gangPrivateOperands().size() > 0)
    printer << " " << ParallelOp::getPrivateKeyword() << "(" <<
        op.gangPrivateOperands() << ")";

  // Gang first private
  if (op.gangFirstPrivateOperands().size() > 0)
    printer << " " << ParallelOp::getFirstPrivateKeyword() << "(" <<
        op.gangFirstPrivateOperands() << ")";
  
  // create
  if (op.createOperands().size() > 0)
    printer << " " << ParallelOp::getCreateKeyword() << "(" <<
        op.createOperands() << ")";

  // copyin
  if (op.copyinOperands().size() > 0)
    printer << " " << ParallelOp::getCopyinKeyword() << "(" <<
        op.copyinOperands() << ")";

  // copyout
  if (op.copyoutOperands().size() > 0)
    printer << " " << ParallelOp::getCopyoutKeyword() << "(" <<
        op.copyoutOperands() << ")";

  if (auto ifCond = op.ifCond())
    printer << " " << ParallelOp::getIfKeyword() << "(" << ifCond << ")";

  printer.printRegion(op.getBody(), false, false);
  printer.printOptionalAttrDictWithKeyword(
      op.getAttrs(), ParallelOp::getOperandSegmentSizeAttr());
}

//===----------------------------------------------------------------------===//
// DataOp
//===----------------------------------------------------------------------===//

// Parse acc.data operation
// operation := `acc.parallel` `present(value-list)?` `copy(value-list)?`
//                             `copyin(value-list)?` `copyout(value-list)?`
//                             `create(value-list)?` `no_create(value-list)`?
//                             `delete(value-list)?` `attach(value-list)`?
//                             `detach(value-list)?`
//                             region attr-dict?
static ParseResult parseDataOp(OpAsmParser &parser, OperationState &result) {
  auto &builder = parser.getBuilder();
  SmallVector<OpAsmParser::OperandType, 8> presentOperands, copyOperands,
      copyinOperands, copyoutOperands, createOperands, noCreateOperands,
      deleteOperands, attachOperands, detachOperands;
  SmallVector<Type, 8> operandsTypes;

  // present()?
  if (failed(parseOperandList(parser, DataOp::getPresentKeyword(),
      presentOperands, operandsTypes, result)))
    return failure();

  // copy()?
  if (failed(parseOperandList(parser, DataOp::getCopyKeyword(),
      copyOperands, operandsTypes, result)))
    return failure();

  // copyin()?
  if (failed(parseOperandList(parser, DataOp::getCopyinKeyword(),
      copyinOperands, operandsTypes, result)))
    return failure();

  // copyout()?
  if (failed(parseOperandList(parser, DataOp::getCopyoutKeyword(),
      copyoutOperands, operandsTypes, result)))
    return failure();

  // create()?
  if (failed(parseOperandList(parser, DataOp::getCreateKeyword(),
      createOperands, operandsTypes, result)))
    return failure();

  // no_create()?
  if (failed(parseOperandList(parser, DataOp::getCreateKeyword(),
      noCreateOperands, operandsTypes, result)))
    return failure();

  // delete()?
  if (failed(parseOperandList(parser, DataOp::getDeleteKeyword(),
      deleteOperands, operandsTypes, result)))
    return failure();

  // attach()?
  if (failed(parseOperandList(parser, DataOp::getAttachKeyword(),
      attachOperands, operandsTypes, result)))
    return failure();

  // detach()?
  if (failed(parseOperandList(parser, DataOp::getDetachKeyword(),
      detachOperands, operandsTypes, result)))
    return failure();

  // Data op region
  if (failed(parseRegions<ParallelOp>(parser, result)))
    return failure();

  result.addAttribute(
      ParallelOp::getOperandSegmentSizeAttr(),
      builder.getI32VectorAttr({static_cast<int32_t>(presentOperands.size()),
                                static_cast<int32_t>(copyOperands.size()),
                                static_cast<int32_t>(copyinOperands.size()),
                                static_cast<int32_t>(copyoutOperands.size()),
                                static_cast<int32_t>(createOperands.size()),
                                static_cast<int32_t>(noCreateOperands.size()),
                                static_cast<int32_t>(deleteOperands.size()),
                                static_cast<int32_t>(attachOperands.size()),
                                static_cast<int32_t>(detachOperands.size())}));

  // Additional attributes
  if (failed(parseOptionalAttributes(parser, result)))
    return failure();

  return success();
}

static void printDataOp(DataOp &op, OpAsmPrinter &printer) {
  printer << DataOp::getOperationName();

  // present list
  if (op.presentOperands().size() > 0)
    printer << " " << DataOp::getPresentKeyword() << "(" <<
        op.presentOperands() << ")";

  // copy list
  if (op.copyOperands().size() > 0)
    printer << " " << DataOp::getCopyKeyword() << "(" <<
        op.copyOperands() << ")";

  // copyin list
  if (op.copyinOperands().size() > 0)
    printer << " " << DataOp::getCopyinKeyword() << "(" <<
        op.copyinOperands() << ")";

  // copyout list
  if (op.copyoutOperands().size() > 0)
    printer << " " << DataOp::getCopyoutKeyword() << "(" <<
        op.copyoutOperands() << ")";

  // create list
  if (op.createOperands().size() > 0)
    printer << " " << DataOp::getCreateKeyword() << "(" <<
        op.createOperands() << ")";

  // no_create list
  if (op.noCreateOperands().size() > 0)
    printer << " " << DataOp::getNoCreateKeyword() << "(" <<
        op.noCreateOperands() << ")";

  // delete list
  if (op.deleteOperands().size() > 0)
    printer << " " << DataOp::getDeleteKeyword() << "(" <<
        op.deleteOperands() << ")";

  // attach list
  if (op.attachOperands().size() > 0)
    printer << " " << DataOp::getAttachKeyword() << "(" <<
        op.attachOperands() << ")";

  // detach list
  if (op.detachOperands().size() > 0)
    printer << " " << DataOp::getDetachKeyword() << "(" <<
        op.detachOperands() << ")";

  printer.printRegion(op.getBody(), false, false);
  printer.printOptionalAttrDictWithKeyword(
      op.getAttrs(), ParallelOp::getOperandSegmentSizeAttr());
}

void ParallelOp::build(OpBuilder &builder, OperationState &state, 
                       ValueRange gangPrivateOperands,
                       ValueRange gangFirstPrivateOperands, 
                       ValueRange createOperands, ValueRange copyInOperands,
                       ValueRange copyOutOperands) {
  ensureTerminator(*state.addRegion(), builder, state.location);
}

//===----------------------------------------------------------------------===//
// LoopOp
//===----------------------------------------------------------------------===//

void LoopOp::build(OpBuilder &builder, OperationState &result) {
  Region *bodyRegion = result.addRegion();
  LoopOp::ensureTerminator(*bodyRegion, builder, result.location);
}

// Parse acc.loop operation
// operation := `acc.loop` `gang?` `vector?` `seq?`
//                         region attr-dict?
static ParseResult parseLoopOp(OpAsmParser &parser, OperationState &state) {

  Builder &builder = parser.getBuilder();
  unsigned executionMapping = 0;
  if (succeeded(parser.parseOptionalKeyword(LoopOp::getGangAttrName())))
    executionMapping |= OpenACCExecMapping::GANG;

  if (succeeded(parser.parseOptionalKeyword(LoopOp::getVectorAttrName())))
    executionMapping |= OpenACCExecMapping::VECTOR;

  if (succeeded(parser.parseOptionalKeyword(LoopOp::getWorkerAttrName())))
    executionMapping |= OpenACCExecMapping::WORKER;

  if (executionMapping != 0)
    state.addAttribute(LoopOp::getExecutionMappingAttrName(),
                       builder.getI64IntegerAttr(executionMapping));

  if (succeeded(parser.parseOptionalKeyword(LoopOp::getSeqAttrName())))
    state.addAttribute(LoopOp::getSeqAttrName(), builder.getUnitAttr());

  // Parse optional results in case there is a reduce.
  if (parser.parseOptionalArrowTypeList(state.types))
    return failure();

  if (failed(parseRegions<LoopOp>(parser, state)))
    return failure();

  if (failed(parseOptionalAttributes(parser, state)))
    return failure();

  return success();
}

static void printLoopOp(LoopOp &op, OpAsmPrinter &printer) {
  printer << LoopOp::getOperationName();
  bool printBlockTerminators = false;

  unsigned execMapping = (op.getAttrOfType<IntegerAttr>(
                              LoopOp::getExecutionMappingAttrName()) != nullptr)
                             ? op.getAttrOfType<IntegerAttr>(
                                     LoopOp::getExecutionMappingAttrName())
                                   .getInt()
                             : 0;
  if ((execMapping & OpenACCExecMapping::GANG) == OpenACCExecMapping::GANG)
    printer << " " << LoopOp::getGangAttrName();

  if ((execMapping & OpenACCExecMapping::WORKER) == OpenACCExecMapping::WORKER)
    printer << " " << LoopOp::getWorkerAttrName();

  if ((execMapping & OpenACCExecMapping::VECTOR) == OpenACCExecMapping::VECTOR)
    printer << " " << LoopOp::getVectorAttrName();

  printFormattedAttr(op, printer, LoopOp::getSeqAttrName(), false);

  if(op.getNumResults() > 0) {
    printer << " -> (" << op.getResultTypes() << ")";
    printBlockTerminators = true;
  }

  printer.printRegion(op.getBody(), false, printBlockTerminators);

  SmallVector<StringRef, 1> formattedAttrs;
  formattedAttrs.push_back(LoopOp::getExecutionMappingAttrName());
  formattedAttrs.push_back(LoopOp::getSeqAttrName());

  printer.printOptionalAttrDictWithKeyword(op.getAttrs(), formattedAttrs);
}

//===----------------------------------------------------------------------===//
// YieldOp
//===----------------------------------------------------------------------===//
static LogicalResult verify(YieldOp op) {
  auto parentOp = op.getParentOp();
  auto results = parentOp->getResults();
  auto operands = op.getOperands();

  if (isa<acc::LoopOp>(parentOp) || isa<acc::ParallelOp>(parentOp)) {
    if (parentOp->getNumResults() != op.getNumOperands())
      return op.emitOpError() << "parent of yield must have same number of "
                                 "results as the yield operands";
    for (auto e : llvm::zip(results, operands)) {
      if (std::get<0>(e).getType() != std::get<1>(e).getType())
        return op.emitOpError()
               << "types mismatch between yield op and its parent";
    }
  } else {
    return op.emitOpError()
           << "yield only terminates Loop or Parallel regions";
  }
  return success();
}

//===----------------------------------------------------------------------===//
// ReductionOp
//===----------------------------------------------------------------------===//

// void ReductionOp::build(OpBuilder &builder, OperationState &result,
//     Value operand) {
//   auto type = operand.getType();
//   result.addOperands(operand);
//   Region *bodyRegion = result.addRegion();

//   Block *b = new Block();
//   b->addArguments(ArrayRef<Type>{type, type});
//   bodyRegion->getBlocks().insert(bodyRegion->end(), b);
// }

static LogicalResult verify(ReductionOp op) {
  // The region of a ReduceOp has two arguments of the same type as its operand.
  // auto type = op.operand().getType();
  // Block &block = op.reductionOperator().front();
  // if (block.empty())
  //   return op.emitOpError("the block inside reduce should not be empty");
  // if (block.getNumArguments() != 2 ||
  //     llvm::any_of(block.getArguments(), [&](const BlockArgument &arg) {
  //       return arg.getType() != type;
  //     }))
  //   return op.emitOpError()
  //          << "expects two arguments to reduce block of type " << type;

  // // Check that the block is terminated by a ReduceReturnOp.
  // if (!isa<ReduceReturnOp>(block.getTerminator()))
  //   return op.emitOpError("the block inside reduce should be terminated with a "
  //                         "'loop.reduce.return' op");

  return success();
}


//===----------------------------------------------------------------------===//
// TableGen'd op method definitions
//===----------------------------------------------------------------------===//

#define GET_OP_CLASSES
#include "mlir/Dialect/OpenACC/OpenACC.cpp.inc"

struct OpenACCLoopEmptyConstructFolder : public OpRewritePattern<acc::LoopOp> {
  using OpRewritePattern<acc::LoopOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(acc::LoopOp loopOp,
                                PatternRewriter &rewriter) const override {
    // Check that the body only contains a terminator.
    if (!llvm::hasSingleElement(loopOp.getBody().front()))
      return failure();
    rewriter.eraseOp(loopOp);
    return success();
  }
};

void acc::LoopOp::getCanonicalizationPatterns(
    OwningRewritePatternList &patterns, MLIRContext *context) {
  patterns.insert<OpenACCLoopEmptyConstructFolder>(context);
}
