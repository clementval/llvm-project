//===- SampleProfileProbe.cpp - Pseudo probe Instrumentation -------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the SampleProfileProber transformation.
//
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/IPO/SampleProfileProbe.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/Analysis/TargetLibraryInfo.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/CFG.h"
#include "llvm/IR/Constant.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DebugInfoMetadata.h"
#include "llvm/IR/GlobalValue.h"
#include "llvm/IR/GlobalVariable.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Instruction.h"
#include "llvm/IR/MDBuilder.h"
#include "llvm/ProfileData/SampleProf.h"
#include "llvm/Support/CRC.h"
#include "llvm/Transforms/Instrumentation.h"
#include "llvm/Transforms/Utils/ModuleUtils.h"
#include <vector>

using namespace llvm;
#define DEBUG_TYPE "sample-profile-probe"

STATISTIC(ArtificialDbgLine,
          "Number of probes that have an artificial debug line");

SampleProfileProber::SampleProfileProber(Function &Func) : F(&Func) {
  BlockProbeIds.clear();
  CallProbeIds.clear();
  LastProbeId = (uint32_t)PseudoProbeReservedId::Last;
  computeProbeIdForBlocks();
  computeProbeIdForCallsites();
}

void SampleProfileProber::computeProbeIdForBlocks() {
  for (auto &BB : *F) {
    BlockProbeIds[&BB] = ++LastProbeId;
  }
}

void SampleProfileProber::computeProbeIdForCallsites() {
  for (auto &BB : *F) {
    for (auto &I : BB) {
      if (!isa<CallBase>(I))
        continue;
      if (isa<IntrinsicInst>(&I))
        continue;
      CallProbeIds[&I] = ++LastProbeId;
    }
  }
}

uint32_t SampleProfileProber::getBlockId(const BasicBlock *BB) const {
  auto I = BlockProbeIds.find(const_cast<BasicBlock *>(BB));
  return I == BlockProbeIds.end() ? 0 : I->second;
}

uint32_t SampleProfileProber::getCallsiteId(const Instruction *Call) const {
  auto Iter = CallProbeIds.find(const_cast<Instruction *>(Call));
  return Iter == CallProbeIds.end() ? 0 : Iter->second;
}

void SampleProfileProber::instrumentOneFunc(Function &F, TargetMachine *TM) {
  Module *M = F.getParent();
  MDBuilder MDB(F.getContext());
  // Compute a GUID without considering the function's linkage type. This is
  // fine since function name is the only key in the profile database.
  uint64_t Guid = Function::getGUID(F.getName());

  // Assign an artificial debug line to a probe that doesn't come with a real
  // line. A probe not having a debug line will get an incomplete inline
  // context. This will cause samples collected on the probe to be counted
  // into the base profile instead of a context profile. The line number
  // itself is not important though.
  auto AssignDebugLoc = [&](Instruction *I) {
    assert((isa<PseudoProbeInst>(I) || isa<CallBase>(I)) &&
           "Expecting pseudo probe or call instructions");
    if (!I->getDebugLoc()) {
      if (auto *SP = F.getSubprogram()) {
        auto DIL = DebugLoc::get(0, 0, SP);
        I->setDebugLoc(DIL);
        ArtificialDbgLine++;
        LLVM_DEBUG({
          dbgs() << "\nIn Function " << F.getName()
                 << " Probe gets an artificial debug line\n";
          I->dump();
        });
      }
    }
  };

  // Probe basic blocks.
  for (auto &I : BlockProbeIds) {
    BasicBlock *BB = I.first;
    uint32_t Index = I.second;
    // Insert a probe before an instruction with a valid debug line number which
    // will be assigned to the probe. The line number will be used later to
    // model the inline context when the probe is inlined into other functions.
    // Debug instructions, phi nodes and lifetime markers do not have an valid
    // line number. Real instructions generated by optimizations may not come
    // with a line number either.
    auto HasValidDbgLine = [](Instruction *J) {
      return !isa<PHINode>(J) && !isa<DbgInfoIntrinsic>(J) &&
             !J->isLifetimeStartOrEnd() && J->getDebugLoc();
    };

    Instruction *J = &*BB->getFirstInsertionPt();
    while (J != BB->getTerminator() && !HasValidDbgLine(J)) {
      J = J->getNextNode();
    }

    IRBuilder<> Builder(J);
    assert(Builder.GetInsertPoint() != BB->end() &&
           "Cannot get the probing point");
    Function *ProbeFn =
        llvm::Intrinsic::getDeclaration(M, Intrinsic::pseudoprobe);
    Value *Args[] = {Builder.getInt64(Guid), Builder.getInt64(Index),
                     Builder.getInt32(0)};
    auto *Probe = Builder.CreateCall(ProbeFn, Args);
    AssignDebugLoc(Probe);
  }

  // Probe both direct calls and indirect calls. Direct calls are probed so that
  // their probe ID can be used as an call site identifier to represent a
  // calling context.
  for (auto &I : CallProbeIds) {
    auto *Call = I.first;
    uint32_t Index = I.second;
    uint32_t Type = cast<CallBase>(Call)->getCalledFunction()
                        ? (uint32_t)PseudoProbeType::DirectCall
                        : (uint32_t)PseudoProbeType::IndirectCall;
    AssignDebugLoc(Call);
    // Levarge the 32-bit discriminator field of debug data to store the ID and
    // type of a callsite probe. This gets rid of the dependency on plumbing a
    // customized metadata through the codegen pipeline.
    uint32_t V = PseudoProbeDwarfDiscriminator::packProbeData(Index, Type);
    if (auto DIL = Call->getDebugLoc()) {
      DIL = DIL->cloneWithDiscriminator(V);
      Call->setDebugLoc(DIL);
    }
  }
}

PreservedAnalyses SampleProfileProbePass::run(Module &M,
                                              ModuleAnalysisManager &AM) {
  for (auto &F : M) {
    if (F.isDeclaration())
      continue;
    SampleProfileProber ProbeManager(F);
    ProbeManager.instrumentOneFunc(F, TM);
  }

  return PreservedAnalyses::none();
}
