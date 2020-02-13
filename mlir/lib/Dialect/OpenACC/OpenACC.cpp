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

static ArrayRef<StringRef> getParallelOpFormattedAttrs() {
  return {ParallelOp::getNumGangsAttrName(), 
          ParallelOp::getNumWorkersAttrName()};
}

static ParseResult parseFormattedAttrs(OpAsmParser &parser, 
                                        OperationState &state, 
                                        StringRef attrName) {
  if(succeeded(parser.parseOptionalKeyword(attrName))) {
    Attribute attr;
    parser.parseLParen();
    parser.parseAttribute(attr, attrName, state.attributes);
    parser.parseRParen();
  }
  return success();
}

// Parse acc.parallel operation
// operation := `acc.parallel` `num_gangs(value)?` `num_workers(value)?`
//                             region attr-dict?
static ParseResult parseParallelOp(OpAsmParser &parser, OperationState &state) {
  if(failed(parseFormattedAttrs(parser, state, 
                                ParallelOp::getNumGangsAttrName())))
    return failure();

  if(failed(parseFormattedAttrs(parser, state, 
                                ParallelOp::getNumWorkersAttrName())))
    return failure();

  if(failed(parseRegions<ParallelOp>(parser, state)))
    return failure();

  if(failed(parseOptionalAttributes(parser, state)))
    return failure();

  return success();
}

static void printFormattedAttributes(Operation *op, OpAsmPrinter &printer, 
                                     ArrayRef<StringRef> formattedAttrs) {
  for(auto attr : op->getAttrs()) {
    if(llvm::is_contained(formattedAttrs, attr.first.strref())) {
      printer << " " << attr.first.strref() << "(";
      printer.printAttribute(attr.second);
      printer << ")";
    }
  }
}

static void printParallelOp(Operation *op, OpAsmPrinter &printer) {
  printer << op->getName();
  
  printFormattedAttributes(op, printer, getParallelOpFormattedAttrs());

  for (auto &region : op->getRegions()) {
    printer.printRegion(region, false, false);
  }

  SmallVector<NamedAttribute, 8> filteredAttrs(
    llvm::make_filter_range(op->getAttrs(), [&](NamedAttribute attr) {
      return !llvm::is_contained(getParallelOpFormattedAttrs(), attr.first.strref());
    }));

  if (!filteredAttrs.empty()) {
    printer.printOptionalAttrDictWithKeyword(op->getAttrs(), getParallelOpFormattedAttrs());
  }
}

static void printRegionOp(Operation *op, OpAsmPrinter &printer) {
  printer << op->getName();

  for (auto &region : op->getRegions()) {
    printer.printRegion(region, false, false);
  }

  if (!op->getAttrs().empty()) {
    printer.printOptionalAttrDictWithKeyword(op->getAttrs());
  }
}

void ParallelOp::build(Builder *builder, OperationState &state) {
  ensureTerminator(*state.addRegion(), *builder, state.location);
}

//===----------------------------------------------------------------------===//
// LoopOp
//===----------------------------------------------------------------------===//
/*
void LoopOp::build(Builder *builder, OperationState &result) {
    Region *bodyRegion = result.addRegion();
}*/

// static void print(OpAsmPrinter &p, LoopOp op) {
//    p << LoopOp::getOperationName();
//}

// TODO use the Region parser for now
/*
static ParseResult parseLoopOp(OpAsmParser &parser, OperationState &result) {
    //auto &builder = parser.getBuilder();
    Region *bodyRegion = result.addRegion();
    if (parser.parseRegion(*bodyRegion, {}, {}))
        return failure();
//    IfOp::ensureTerminator(*thenRegion, parser.getBuilder(), result.location);
    // Parse the optional attribute list.
    if (parser.parseOptionalAttrDict(result.attributes))
        return failure();
    return success();
}*/

/*
static void print(OpAsmPrinter &p, IfOp op) {
    p << LoopOp::getOperationName();
    //p.printRegion(op.bodyRegion(),
            */
/*printEntryBlockArgs=*/  /*false,
                           */
/*printBlockTerminators=*//*false);
p.printOptionalAttrDict(op.getAttrs());
}*/

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
