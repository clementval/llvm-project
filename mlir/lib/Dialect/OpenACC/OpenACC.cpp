//===- OpenACC.cpp - OpenACC MLIR Operations ------------------------------===//
//
// Part of the MLIR Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// =============================================================================

#include "mlir/Dialect/OpenACC/OpenACC.h"
#include "mlir/Dialect/StandardOps/Ops.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Function.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/Module.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/StandardTypes.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/MathExtras.h"
#include "mlir/Support/STLExtras.h"
#include "mlir/Transforms/SideEffectsInterface.h"

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

/*
template <typename StructureOp>
static ParseResult parseRegionOp(OpAsmParser &parser, OperationState &state,
                                 unsigned int nRegions = 1) {
    llvm::SmallVector<Region *, 2> regions;
    for (unsigned int i = 0; i < nRegions; ++i)
        regions.push_back(state.addRegion());
    for (auto &region : regions) {
        if (parser.parseRegion(*region, */
/*arguments=*//*
{}, */
/*argTypes=*/ /*
 {}))
             return failure();
         StructureOp::ensureTerminator(*region, parser.getBuilder(),
 state.location);
     }
     if (succeeded(parser.parseOptionalKeyword("attributes"))) {
         if (parser.parseOptionalAttrDict(state.attributes))
             return failure();
     }
     return success();
 }*/

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

static void printFormattedAttr(Operation *op, OpAsmPrinter &printer,
                               StringRef attrName, bool hasValue = true) {
  if (op->getAttr(attrName) != nullptr) {
    printer << " " << attrName;
    if (hasValue) {
      printer << "(";
      printer.printAttribute(op->getAttr(attrName));
      printer << ")";
    }
  }
}

//===----------------------------------------------------------------------===//
// ParallelOp
//===----------------------------------------------------------------------===//

// Parse acc.parallel operation
// operation := `acc.parallel` `num_gangs(value)?` `num_workers(value)?`
//                             region attr-dict?
static ParseResult parseParallelOp(OpAsmParser &parser, OperationState &state) {
  if (failed(
          parseFormattedAttr(parser, state, ParallelOp::getNumGangsAttrName())))
    return failure();

  if (failed(parseFormattedAttr(parser, state,
                                ParallelOp::getNumWorkersAttrName())))
    return failure();

  if (failed(parseRegions<ParallelOp>(parser, state)))
    return failure();

  if (failed(parseOptionalAttributes(parser, state)))
    return failure();

  return success();
}

static void printParallelOp(Operation *op, OpAsmPrinter &printer) {
  printer << op->getName();

  printFormattedAttr(op, printer, ParallelOp::getNumGangsAttrName());
  printFormattedAttr(op, printer, ParallelOp::getNumWorkersAttrName());

  SmallVector<StringRef, 2> formattedAttrs;
  formattedAttrs.push_back(ParallelOp::getNumGangsAttrName());
  formattedAttrs.push_back(ParallelOp::getNumWorkersAttrName());

  for (auto &region : op->getRegions()) {
    printer.printRegion(region, false, false);
  }

  SmallVector<NamedAttribute, 8> filteredAttrs(
      llvm::make_filter_range(op->getAttrs(), [&](NamedAttribute attr) {
        return !llvm::is_contained(formattedAttrs, attr.first.strref());
      }));

  if (!filteredAttrs.empty()) {
    printer.printOptionalAttrDictWithKeyword(op->getAttrs(), formattedAttrs);
  }
}

void ParallelOp::build(Builder *builder, OperationState &state) {
  ensureTerminator(*state.addRegion(), *builder, state.location);
}

//===----------------------------------------------------------------------===//
// LoopOp
//===----------------------------------------------------------------------===//

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

  if (executionMapping != 0)
    state.addAttribute(LoopOp::getExecutionMappingAttrName(),
                       builder.getI64IntegerAttr(executionMapping));

  if (succeeded(parser.parseOptionalKeyword(LoopOp::getSeqAttrName())))
    state.addAttribute(LoopOp::getSeqAttrName(), builder.getUnitAttr());

  if (failed(parseRegions<LoopOp>(parser, state)))
    return failure();

  if (failed(parseOptionalAttributes(parser, state)))
    return failure();

  return success();
}

static void printLoopOp(Operation *op, OpAsmPrinter &printer) {
  printer << op->getName();

  unsigned execMapping = (op->getAttrOfType<IntegerAttr>(
                              LoopOp::getExecutionMappingAttrName()) != nullptr)
                             ? op->getAttrOfType<IntegerAttr>(
                                     LoopOp::getExecutionMappingAttrName())
                                   .getInt()
                             : 0;
  if ((execMapping & OpenACCExecMapping::GANG) == OpenACCExecMapping::GANG) {
    printer << " " << LoopOp::getGangAttrName();
  }
  if ((execMapping & OpenACCExecMapping::VECTOR) ==
      OpenACCExecMapping::VECTOR) {
    printer << " " << LoopOp::getVectorAttrName();
  }

  printFormattedAttr(op, printer, LoopOp::getSeqAttrName(), false);

  for (auto &region : op->getRegions()) {
    printer.printRegion(region, false, false);
  }

  SmallVector<StringRef, 1> formattedAttrs;
  formattedAttrs.push_back(LoopOp::getExecutionMappingAttrName());
  formattedAttrs.push_back(LoopOp::getSeqAttrName());

  printer.printOptionalAttrDictWithKeyword(op->getAttrs(), formattedAttrs);
}

//===----------------------------------------------------------------------===//
// TableGen'd op method definitions
//===----------------------------------------------------------------------===//

#define GET_OP_CLASSES
#include "mlir/Dialect/OpenACC/OpenACC.cpp.inc"

struct OpenACCLoopEmptyConstructFolder : public OpRewritePattern<acc::LoopOp> {
  using OpRewritePattern<acc::LoopOp>::OpRewritePattern;

  PatternMatchResult matchAndRewrite(acc::LoopOp loopOp,
                                     PatternRewriter &rewriter) const override {
    // Check that the body only contains a terminator.
    if (!has_single_element(loopOp.getBody()))
      return matchFailure();
    rewriter.eraseOp(loopOp);
    return matchSuccess();
  }
};

void acc::LoopOp::getCanonicalizationPatterns(
    OwningRewritePatternList &patterns, MLIRContext *context) {
  patterns.insert<OpenACCLoopEmptyConstructFolder>(context);
}
