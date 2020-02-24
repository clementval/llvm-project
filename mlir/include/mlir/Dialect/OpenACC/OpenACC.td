//===- OpenACC.td - OpenACC operation definitions ----------*- tablegen -*-===//
//
// Part of the MLIR Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// =============================================================================
//
// Defines MLIR OpenACC operations.
//
//===----------------------------------------------------------------------===//

#ifndef OPENACC_TD
#define OPENACC_TD

#ifndef OP_BASE
include "mlir/IR/OpBase.td"
#endif // OP_BASE

def OpenACC_Dialect : Dialect {
    let name = "acc";

    let description = [{
        An OpenACC dialect for MLIR.

        This dialect models the construct from the OpenACC directive language.
    }];

  let cppNamespace = "acc";
}

// Base class for Loop dialect ops.
class OpenACC_Op<string mnemonic, list<OpTrait> traits = []> :
    Op<OpenACC_Dialect, mnemonic, traits> {

    // For every standard op, there needs to be a:
    //   * void print(OpAsmPrinter &p, ${C++ class of Op} op)
    //   * LogicalResult verify(${C++ class of Op} op)
    //   * ParseResult parse${C++ class of Op}(OpAsmParser &parser,
    //                                         OperationState &result)
    // functions.
    let printer = [{ return ::print(p, *this); }];
    let verifier = [{ return ::verify(*this); }];
    let parser = [{ return ::parse$cppClass(parser, result); }];
}

class OpenACC_ScopeVerifier<string opName> : PredOpTrait<
  "op must appear in a '" # opName #"' block",
  CPred<"llvm::isa_and_nonnull<" # opName # ">($_op.getParentOp())">>;

// Require regions to have openacc terminator.
def ImplicitOpenACCTerminator
    : SingleBlockImplicitTerminator<"OpenACCTerminatorOp">;

def InParallelScope : OpenACC_ScopeVerifier<"ParallelOp">;

def ParallelOp : OpenACC_Op<"parallel",
                            [SingleBlockImplicitTerminator<"ParallelEndOp">]> {
    let summary = "parallel construct";
    let description = [{
        The "acc.parallel" operation represents a parallel construct block. It has
        one region capturing the parallel block.

        acc.parallel {
           ... // body
        }
    }];

    let builders = [
      OpBuilder<"Builder *, OperationState &">
    ];
    let parser = [{ return parseParallelOp(parser, result); }];
    let printer = [{ printParallelOp(*this, p); }];

    let arguments = (ins Variadic<AnyType>:$privateArgs);
    let regions = (region SizedRegion<1>:$region);
    let verifier = [{ return success(); }];
    let skipDefaultBuilders = 1;

    let extraClassDeclaration = [{

        Region &getBody() { return this->getOperation()->getRegion(0); }

        static StringRef getNumGangsAttrName() { return "num_gangs"; }
        static StringRef getNumWorkersAttrName() { return "num_workers"; }
        static StringRef getPrivateKeyword() { return "private"; }
        static StringRef getNumGangPrivateAttrName() {
          return "gang_private";
        }

        /// Returns the number of gang private.
        unsigned getNumGangPrivates() {
          return getAttrOfType<IntegerAttr>(getNumGangPrivateAttrName())
              .getInt();
        }

        iterator_range<Operation::operand_iterator> getGangPrivates() {
          return getOperands().take_front(getNumGangPrivates());
        }

        /// Returns the number of gangs defined for the parallel op
        unsigned getNumGangs() {
          return getAttrOfType<IntegerAttr>(getNumGangsAttrName()) != nullptr 
              ? getAttrOfType<IntegerAttr>(getNumGangsAttrName()).getInt() 
              : 1;
        }

        /// Returns the number of workers defined for the parallel op
        unsigned getNumWorkers() {
          return 
              getAttrOfType<IntegerAttr>(getNumWorkersAttrName()) != nullptr
              ? getAttrOfType<IntegerAttr>(getNumWorkersAttrName()).getInt()
              : 1;
        }

        
    }];

}

def ParallelEndOp : OpenACC_Op<"_parallel_end", [Terminator]> {
    let arguments = (ins);
    let results = (outs);

    let parser = [{ return parseNoIOOp(parser, result); }];
    let printer = [{ printNoIOOp(getOperation(), p); }];
    let verifier = [{ return success(); }];
}

def LoopOp : OpenACC_Op<"loop",
                        [SingleBlockImplicitTerminator<"LoopEndOp">]> {
    let summary = "loop construct";
    let description = [{
        The "acc.loop"
    }];

    let builders = [OpBuilder<"Builder *, OperationState &">];

    let parser = [{ return parseLoopOp(parser, result); }];
    let printer = [{ printLoopOp(*this, p); }];

    let regions = (region SizedRegion<1>:$region);
    let verifier = [{ return success(); }];
    let skipDefaultBuilders = 1;
    let hasCanonicalizer = 1;

    let extraClassDeclaration = [{
        Region &getBody() { return this->getOperation()->getRegion(0); }

        static StringRef getExecutionMappingAttrName() { return "exec_mapping"; }
        static StringRef getCollapseAttrName() { return "collapse"; }
        static StringRef getGangAttrName() { return "gang"; }
        static StringRef getSeqAttrName() { return "seq"; }
        static StringRef getVectorAttrName() { return "vector"; }
        static StringRef getWorkerAttrName() { return "worker"; }

        bool hasCollapseAttr() {
            return getAttrOfType<IntegerAttr>(getCollapseAttrName()) != nullptr;
        }

        unsigned getCollapse() {
          return 
              getAttrOfType<IntegerAttr>(getCollapseAttrName()) != nullptr
              ? getAttrOfType<IntegerAttr>(getCollapseAttrName()).getInt()
              : 1;
        }

        unsigned getExecutionMappingAttr() {
          return 
              getAttrOfType<IntegerAttr>(getExecutionMappingAttrName()) != nullptr
              ? getAttrOfType<IntegerAttr>(getExecutionMappingAttrName()).getInt()
              : 0;
        }

        bool isGangVector() {
          unsigned execMapping = getExecutionMappingAttr();
          return (execMapping & OpenACCExecMapping::GANG_VECTOR) 
              == OpenACCExecMapping::GANG_VECTOR 
              || execMapping == OpenACCExecMapping::NONE;
        }
        
        bool isGang() {
          unsigned execMapping = getExecutionMappingAttr();
          return (execMapping & OpenACCExecMapping::GANG) == OpenACCExecMapping::GANG;
        }

        bool isVector() {
          unsigned execMapping = getExecutionMappingAttr();
          return (execMapping & OpenACCExecMapping::VECTOR) == OpenACCExecMapping::VECTOR;
        }

        bool isWorker() {
          unsigned execMapping = getExecutionMappingAttr();
          return (execMapping & OpenACCExecMapping::WORKER) == OpenACCExecMapping::WORKER;
        }

        bool hasSeqAttr() {
          UnitAttr seqAttr = getAttrOfType<UnitAttr>(getSeqAttrName());
          return static_cast<bool>(seqAttr);
        }

    }];
}

def LoopEndOp : OpenACC_Op<"_loop_end", [Terminator]> {
    let arguments = (ins);
    let results = (outs);

    let parser = [{ return parseNoIOOp(parser, result); }];
    let printer = [{ printNoIOOp(getOperation(), p); }];
    let verifier = [{ return success(); }];
}

#endif // OPENACC_TD

