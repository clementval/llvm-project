// RUN: llvm-tblgen -gen-directive-decl -I %p/../../include %s | FileCheck -match-full-lines %s
// RUN: llvm-tblgen -gen-directive-gen -I %p/../../include %s | FileCheck -match-full-lines %s -check-prefix=GEN

include "llvm/Frontend/Directive/DirectiveBase.td"

def TestDirectiveLanguage : DirectiveLanguage {
  let name = "Tdl";

  let cppNamespace = "tdl";
  let directivePrefix = "TDLD_";
  let clausePrefix = "TDLC_";
  let makeEnumAvailableInNamespace = 1;
  let enableBitmaskEnumInNamespace = 1;
  let flangClauseBaseClass = "TdlClause";
}

def TDLCV_vala : ClauseVal<"vala",1,1> {}
def TDLCV_valb : ClauseVal<"valb",2,1> {}
def TDLCV_valc : ClauseVal<"valc",3,0> { let isDefault = 1; }

def TDLC_ClauseA : Clause<"clausea"> {
  let flangClass = "TdlClauseA";
  let enumClauseValue = "AKind";
  let allowedClauseValues = [
    TDLCV_vala,
    TDLCV_valb,
    TDLCV_valc
  ];
}

def TDLC_ClauseB : Clause<"clauseb"> {
  let flangClassValue = "IntExpr";
  let isValueOptional = 1;
  let isDefault = 1;
}

def TDL_DirA : Directive<"dira"> {
  let allowedClauses = [
    VersionedClause<TDLC_ClauseA>,
    VersionedClause<TDLC_ClauseB>
  ];
  let isDefault = 1;
}

// CHECK:       #ifndef LLVM_Tdl_INC
// CHECK-NEXT:  #define LLVM_Tdl_INC
// CHECK-EMPTY:
// CHECK-NEXT:  #include "llvm/ADT/BitmaskEnum.h"
// CHECK-EMPTY:
// CHECK-NEXT:  namespace llvm {
// CHECK-NEXT:  class StringRef;
// CHECK-NEXT:  namespace tdl {
// CHECK-EMPTY:
// CHECK-NEXT:  LLVM_ENABLE_BITMASK_ENUMS_IN_NAMESPACE();
// CHECK-EMPTY:
// CHECK-NEXT:  enum class Directive {
// CHECK-NEXT:    TDLD_dira,
// CHECK-NEXT:  };
// CHECK-EMPTY:
// CHECK-NEXT:  static constexpr std::size_t Directive_enumSize = 1;
// CHECK-EMPTY:
// CHECK-NEXT:  constexpr auto TDLD_dira = llvm::tdl::Directive::TDLD_dira;
// CHECK-EMPTY:
// CHECK-NEXT:  enum class Clause {
// CHECK-NEXT:    TDLC_clausea,
// CHECK-NEXT:    TDLC_clauseb,
// CHECK-NEXT:  };
// CHECK-EMPTY:
// CHECK-NEXT:  static constexpr std::size_t Clause_enumSize = 2;
// CHECK-EMPTY:
// CHECK-NEXT:  constexpr auto TDLC_clausea = llvm::tdl::Clause::TDLC_clausea;
// CHECK-NEXT:  constexpr auto TDLC_clauseb = llvm::tdl::Clause::TDLC_clauseb;
// CHECK-EMPTY:
// CHECK-NEXT:  enum class AKind {
// CHECK-NEXT:    TDLCV_vala=1,
// CHECK-NEXT:    TDLCV_valb=2,
// CHECK-NEXT:    TDLCV_valc=3,
// CHECK-NEXT:  };
// CHECK-EMPTY:
// CHECK-NEXT:  constexpr auto TDLCV_vala = llvm::tdl::AKind::TDLCV_vala;
// CHECK-NEXT:  constexpr auto TDLCV_valb = llvm::tdl::AKind::TDLCV_valb;
// CHECK-NEXT:  constexpr auto TDLCV_valc = llvm::tdl::AKind::TDLCV_valc;
// CHECK-EMPTY:
// CHECK-NEXT:  // Enumeration helper functions
// CHECK-NEXT:  Directive getTdlDirectiveKind(llvm::StringRef Str);
// CHECK-EMPTY:
// CHECK-NEXT:  llvm::StringRef getTdlDirectiveName(Directive D);
// CHECK-EMPTY:
// CHECK-NEXT:  Clause getTdlClauseKind(llvm::StringRef Str);
// CHECK-EMPTY:
// CHECK-NEXT:  llvm::StringRef getTdlClauseName(Clause C);
// CHECK-EMPTY:
// CHECK-NEXT:  /// Return true if \p C is a valid clause for \p D in version \p Version.
// CHECK-NEXT:  bool isAllowedClauseForDirective(Directive D, Clause C, unsigned Version);
// CHECK-EMPTY:
// CHECK-NEXT:  AKind getAKind(StringRef);
// CHECK-EMPTY:
// CHECK-NEXT:  } // namespace tdl
// CHECK-NEXT:  } // namespace llvm
// CHECK-NEXT:  #endif // LLVM_Tdl_INC


// GEN:       #ifdef GEN_FLANG_DIRECTIVE_CLAUSE_SETS
// GEN-NEXT:  #undef GEN_FLANG_DIRECTIVE_CLAUSE_SETS
// GEN-EMPTY:
// GEN-NEXT:  namespace llvm {
// GEN-NEXT:  namespace tdl {
// GEN-EMPTY:
// GEN-NEXT:    // Sets for dira
// GEN-EMPTY:
// GEN-NEXT:    static  allowedClauses_TDLD_dira {
// GEN-NEXT:      llvm::tdl::Clause::TDLC_clausea,
// GEN-NEXT:      llvm::tdl::Clause::TDLC_clauseb,
// GEN-NEXT:    };
// GEN-EMPTY:
// GEN-NEXT:    static  allowedOnceClauses_TDLD_dira {
// GEN-NEXT:    };
// GEN-EMPTY:
// GEN-NEXT:    static  allowedExclusiveClauses_TDLD_dira {
// GEN-NEXT:    };
// GEN-EMPTY:
// GEN-NEXT:    static  requiredClauses_TDLD_dira {
// GEN-NEXT:    };
// GEN-NEXT:  } // namespace tdl
// GEN-NEXT:  } // namespace llvm
// GEN-EMPTY:
// GEN-NEXT:  #endif // GEN_FLANG_DIRECTIVE_CLAUSE_SETS
// GEN-EMPTY:
// GEN-NEXT:  #ifdef GEN_FLANG_DIRECTIVE_CLAUSE_MAP
// GEN-NEXT:  #undef GEN_FLANG_DIRECTIVE_CLAUSE_MAP
// GEN-EMPTY:
// GEN-NEXT:  {
// GEN-NEXT:    {llvm::tdl::Directive::TDLD_dira,
// GEN-NEXT:      {
// GEN-NEXT:        llvm::tdl::allowedClauses_TDLD_dira,
// GEN-NEXT:        llvm::tdl::allowedOnceClauses_TDLD_dira,
// GEN-NEXT:        llvm::tdl::allowedExclusiveClauses_TDLD_dira,
// GEN-NEXT:        llvm::tdl::requiredClauses_TDLD_dira,
// GEN-NEXT:      }
// GEN-NEXT:    },
// GEN-NEXT:  }
// GEN-EMPTY:
// GEN-NEXT:  #endif // GEN_FLANG_DIRECTIVE_CLAUSE_MAP
// GEN-EMPTY:
// GEN-NEXT:  #ifdef GEN_FLANG_CLAUSE_PARSER_CLASSES
// GEN-NEXT:  #undef GEN_FLANG_CLAUSE_PARSER_CLASSES
// GEN-EMPTY:
// GEN-NEXT:  WRAPPER_CLASS(Clauseb, std::optional<IntExpr>);
// GEN-EMPTY:
// GEN-NEXT:  #endif // GEN_FLANG_CLAUSE_PARSER_CLASSES
// GEN-EMPTY:
// GEN-NEXT:  #ifdef GEN_FLANG_CLAUSE_PARSER_CLASSES_LIST
// GEN-NEXT:  #undef GEN_FLANG_CLAUSE_PARSER_CLASSES_LIST
// GEN-EMPTY:
// GEN-NEXT:  TdlClauseA
// GEN-NEXT:  , Clauseb
// GEN-EMPTY:
// GEN-NEXT:  #endif // GEN_FLANG_CLAUSE_PARSER_CLASSES_LIST
// GEN-EMPTY:
// GEN-NEXT:  #ifdef GEN_FLANG_DUMP_PARSE_TREE_CLAUSES
// GEN-NEXT:  #undef GEN_FLANG_DUMP_PARSE_TREE_CLAUSES
// GEN-EMPTY:
// GEN-NEXT:  NODE(TdlClause, Clauseb)
// GEN-EMPTY:
// GEN-NEXT:  #endif // GEN_FLANG_DUMP_PARSE_TREE_CLAUSES
// GEN-EMPTY:
// GEN-NEXT:  #ifdef GEN_FLANG_CLAUSE_UNPARSE
// GEN-NEXT:  #undef GEN_FLANG_CLAUSE_UNPARSE
// GEN-EMPTY:
// GEN-NEXT:  void Unparse(const TdlClause::Clauseb &x) {
// GEN-NEXT:    Word("CLAUSEB");
// GEN-NEXT:    Walk("(", x.v, ")");
// GEN-NEXT:  }
// GEN-EMPTY:
// GEN-NEXT:  #endif // GEN_FLANG_CLAUSE_UNPARSE

// GEN:       #ifdef GEN_DIRECTIVES_IMPL
// GEN-NEXT:  #undef GEN_DIRECTIVES_IMPL
// GEN-EMPTY:
// GEN-NEXT:  Directive llvm::tdl::getTdlDirectiveKind(llvm::StringRef Str) {
// GEN-NEXT:    return llvm::StringSwitch<Directive>(Str)
// GEN-NEXT:      .Case("dira",TDLD_dira)
// GEN-NEXT:      .Default(TDLD_dira);
// GEN-NEXT:  }
// GEN-EMPTY:
// GEN-NEXT:  llvm::StringRef llvm::tdl::getTdlDirectiveName(Directive Kind) {
// GEN-NEXT:    switch (Kind) {
// GEN-NEXT:      case TDLD_dira:
// GEN-NEXT:        return "dira";
// GEN-NEXT:    }
// GEN-NEXT:    llvm_unreachable("Invalid Tdl Directive kind");
// GEN-NEXT:  }
// GEN-EMPTY:
// GEN-NEXT:  Clause llvm::tdl::getTdlClauseKind(llvm::StringRef Str) {
// GEN-NEXT:    return llvm::StringSwitch<Clause>(Str)
// GEN-NEXT:      .Case("clausea",TDLC_clausea)
// GEN-NEXT:      .Case("clauseb",TDLC_clauseb)
// GEN-NEXT:      .Default(TDLC_clauseb);
// GEN-NEXT:  }
// GEN-EMPTY:
// GEN-NEXT:  llvm::StringRef llvm::tdl::getTdlClauseName(Clause Kind) {
// GEN-NEXT:    switch (Kind) {
// GEN-NEXT:      case TDLC_clausea:
// GEN-NEXT:        return "clausea";
// GEN-NEXT:      case TDLC_clauseb:
// GEN-NEXT:        return "clauseb";
// GEN-NEXT:    }
// GEN-NEXT:    llvm_unreachable("Invalid Tdl Clause kind");
// GEN-NEXT:  }
// GEN-EMPTY:
// GEN-NEXT:  AKind llvm::tdl::getAKind(llvm::StringRef Str) {
// GEN-NEXT:    return llvm::StringSwitch<AKind>(Str)
// GEN-NEXT:      .Case("vala",TDLCV_vala)
// GEN-NEXT:      .Case("valb",TDLCV_valb)
// GEN-NEXT:      .Case("valc",TDLCV_valc)
// GEN-NEXT:      .Default(TDLCV_valc);
// GEN-NEXT:  }
// GEN-EMPTY:
// GEN-NEXT:  bool llvm::tdl::isAllowedClauseForDirective(Directive D, Clause C, unsigned Version) {
// GEN-NEXT:    assert(unsigned(D) <= llvm::tdl::Directive_enumSize);
// GEN-NEXT:    assert(unsigned(C) <= llvm::tdl::Clause_enumSize);
// GEN-NEXT:    switch (D) {
// GEN-NEXT:      case TDLD_dira:
// GEN-NEXT:        switch (C) {
// GEN-NEXT:          case TDLC_clausea:
// GEN-NEXT:            return 1 <= Version && 2147483647 >= Version;
// GEN-NEXT:          case TDLC_clauseb:
// GEN-NEXT:            return 1 <= Version && 2147483647 >= Version;
// GEN-NEXT:          default:
// GEN-NEXT:            return false;
// GEN-NEXT:        }
// GEN-NEXT:        break;
// GEN-NEXT:    }
// GEN-NEXT:    llvm_unreachable("Invalid Tdl Directive kind");
// GEN-NEXT:  }
// GEN-EMPTY:
// GEN-NEXT:  #endif // GEN_DIRECTIVES_IMPL