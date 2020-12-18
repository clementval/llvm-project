// RUN: llvm-tblgen -gen-directive-decl -I %p/../../include %s | FileCheck -match-full-lines %s
// RUN: llvm-tblgen -gen-directive-gen -I %p/../../include %s | FileCheck -match-full-lines %s -check-prefix=GEN

include "llvm/Frontend/Directive/DirectiveBase.td"

def TestDirectiveLanguage : DirectiveLanguage {
  let name = "Tdl";

  let cppNamespace = "tdl";
  let directivePrefix = "TDLD_";
  let clausePrefix = "TDLC_";
  let includeHeader = "tdl.h.inc";
  let flangClauseBaseClass = "TdlClause";
}

def TDLC_ClauseA : Clause<"clausea"> {
  let isImplicit = 1;
}
def TDLC_ClauseB : Clause<"clauseb"> {
  let isDefault = 1;
  let flangClassValue = "IntExpr";
  let isValueList = 1;
}
def TDLC_ClauseC : Clause<"clausec"> {
  let clangClass = "ClauseC";
  let flangClassValue = "Name";
  let defaultValue = "*";
  let isValueOptional = 1;
}
def TDLC_ClauseD : Clause<"claused"> {
  let clangClass = "ClauseD";
  let isImplicit = 1;
}

def TDL_DirA : Directive<"dira"> {
  let allowedClauses = [
    VersionedClause<TDLC_ClauseA, 2, 4>,
    VersionedClause<TDLC_ClauseB, 2>
  ];
  let isDefault = 1;
}

// CHECK:       #ifndef LLVM_Tdl_INC
// CHECK-NEXT:  #define LLVM_Tdl_INC
// CHECK-EMPTY:
// CHECK-NEXT:  namespace llvm {
// CHECK-NEXT:  class StringRef;
// CHECK-NEXT:  namespace tdl {
// CHECK-EMPTY:
// CHECK-NEXT:  enum class Directive {
// CHECK-NEXT:    TDLD_dira,
// CHECK-NEXT:  };
// CHECK-EMPTY:
// CHECK-NEXT:  static constexpr std::size_t Directive_enumSize = 1;
// CHECK-EMPTY:
// CHECK-NEXT:  enum class Clause {
// CHECK-NEXT:    TDLC_clausea,
// CHECK-NEXT:    TDLC_clauseb,
// CHECK-NEXT:    TDLC_clausec,
// CHECK-NEXT:    TDLC_claused,
// CHECK-NEXT:  };
// CHECK-EMPTY:
// CHECK-NEXT:  static constexpr std::size_t Clause_enumSize = 4;
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
// CHECK-NEXT:  } // namespace tdl
// CHECK-NEXT:  } // namespace llvm
// CHECK-NEXT:  #endif // LLVM_Tdl_INC

// GEN:      #ifdef GEN_FLANG_DIRECTIVE_CLAUSE_SETS
// GEN-NEXT: #undef GEN_FLANG_DIRECTIVE_CLAUSE_SETS
// GEN-EMPTY:
// GEN-NEXT: namespace llvm {
// GEN-NEXT: namespace tdl {
// GEN-EMPTY:
// GEN-NEXT:   // Sets for dira
// GEN-EMPTY:
// GEN-NEXT:   static  allowedClauses_TDLD_dira {
// GEN-NEXT:     llvm::tdl::Clause::TDLC_clausea,
// GEN-NEXT:     llvm::tdl::Clause::TDLC_clauseb,
// GEN-NEXT:   };
// GEN-EMPTY:
// GEN-NEXT:   static  allowedOnceClauses_TDLD_dira {
// GEN-NEXT:   };
// GEN-EMPTY:
// GEN-NEXT:   static  allowedExclusiveClauses_TDLD_dira {
// GEN-NEXT:   };
// GEN-EMPTY:
// GEN-NEXT:   static  requiredClauses_TDLD_dira {
// GEN-NEXT:   };
// GEN-NEXT: } // namespace tdl
// GEN-NEXT: } // namespace llvm
// GEN-EMPTY:
// GEN-NEXT: #endif // GEN_FLANG_DIRECTIVE_CLAUSE_SETS
// GEN-EMPTY:
// GEN-NEXT: #ifdef GEN_FLANG_DIRECTIVE_CLAUSE_MAP
// GEN-NEXT: #undef GEN_FLANG_DIRECTIVE_CLAUSE_MAP
// GEN-EMPTY:
// GEN-NEXT: {
// GEN-NEXT:   {llvm::tdl::Directive::TDLD_dira,
// GEN-NEXT:     {
// GEN-NEXT:       llvm::tdl::allowedClauses_TDLD_dira,
// GEN-NEXT:       llvm::tdl::allowedOnceClauses_TDLD_dira,
// GEN-NEXT:       llvm::tdl::allowedExclusiveClauses_TDLD_dira,
// GEN-NEXT:       llvm::tdl::requiredClauses_TDLD_dira,
// GEN-NEXT:     }
// GEN-NEXT:   },
// GEN-NEXT: }
// GEN-EMPTY:
// GEN-NEXT: #endif // GEN_FLANG_DIRECTIVE_CLAUSE_MAP
// GEN-EMPTY:
// GEN-NEXT:  #ifdef GEN_FLANG_CLAUSE_PARSER_CLASSES
// GEN-NEXT:  #undef GEN_FLANG_CLAUSE_PARSER_CLASSES
// GEN-EMPTY:
// GEN-NEXT:  EMPTY_CLASS(Clausea);
// GEN-NEXT:  WRAPPER_CLASS(Clauseb, std::list<IntExpr>);
// GEN-NEXT:  WRAPPER_CLASS(Clausec, std::optional<Name>);
// GEN-NEXT:  EMPTY_CLASS(Claused);
// GEN-EMPTY:
// GEN-NEXT:  #endif // GEN_FLANG_CLAUSE_PARSER_CLASSES
// GEN-EMPTY:
// GEN-NEXT:  #ifdef GEN_FLANG_CLAUSE_PARSER_CLASSES_LIST
// GEN-NEXT:  #undef GEN_FLANG_CLAUSE_PARSER_CLASSES_LIST
// GEN-EMPTY:
// GEN-NEXT:  Clausea
// GEN-NEXT:  , Clauseb
// GEN-NEXT:  , Clausec
// GEN-NEXT:  , Claused
// GEN-EMPTY:
// GEN-NEXT:  #endif // GEN_FLANG_CLAUSE_PARSER_CLASSES_LIST
// GEN-EMPTY:
// GEN-NEXT:  #ifdef GEN_FLANG_DUMP_PARSE_TREE_CLAUSES
// GEN-NEXT:  #undef GEN_FLANG_DUMP_PARSE_TREE_CLAUSES
// GEN-EMPTY:
// GEN-NEXT:  NODE(TdlClause, Clausea)
// GEN-NEXT:  NODE(TdlClause, Clauseb)
// GEN-NEXT:  NODE(TdlClause, Clausec)
// GEN-NEXT:  NODE(TdlClause, Claused)
// GEN-EMPTY:
// GEN-NEXT:  #endif // GEN_FLANG_DUMP_PARSE_TREE_CLAUSES
// GEN-EMPTY:
// GEN-NEXT:  #ifdef GEN_FLANG_CLAUSE_UNPARSE
// GEN-NEXT:  #undef GEN_FLANG_CLAUSE_UNPARSE
// GEN-EMPTY:
// GEN-NEXT:  void Before(const TdlClause::Clausea &) { Word("CLAUSEA"); }
// GEN-NEXT:  void Unparse(const TdlClause::Clauseb &x) {
// GEN-NEXT:    Word("CLAUSEB");
// GEN-NEXT:    Put("(");
// GEN-NEXT:    Walk(x.v, ",");
// GEN-NEXT:    Put(")");
// GEN-NEXT:  }
// GEN-NEXT:  void Unparse(const TdlClause::Clausec &x) {
// GEN-NEXT:  Word("CLAUSEC");
// GEN-NEXT:    Put("(");
// GEN-NEXT:    if (x.v.has_value())
// GEN-NEXT:    Walk(x.v);
// GEN-NEXT:    else
// GEN-NEXT:    Put("*");
// GEN-NEXT:    Put(")");
// GEN-NEXT:  }
// GEN-NEXT:  void Before(const TdlClause::Claused &) { Word("CLAUSED"); }
// GEN-EMPTY:
// GEN-NEXT:  #endif // GEN_FLANG_CLAUSE_UNPARSE

// GEN:       #ifdef GEN_CLANG_CLAUSE_CLASS
// GEN-NEXT:  #undef GEN_CLANG_CLAUSE_CLASS
// GEN-EMPTY:
// GEN-NEXT:  #ifndef CLAUSE
// GEN-NEXT:  #define CLAUSE(Enum, Str, Implicit)
// GEN-NEXT:  #endif
// GEN-NEXT:  #ifndef CLAUSE_CLASS
// GEN-NEXT:  #define CLAUSE_CLASS(Enum, Str, Class)
// GEN-NEXT:  #endif
// GEN-NEXT:  #ifndef CLAUSE_NO_CLASS
// GEN-NEXT:  #define CLAUSE_NO_CLASS(Enum, Str)
// GEN-NEXT:  #endif
// GEN-EMPTY:
// GEN-NEXT:  #define __CLAUSE(Name, Class)                      \
// GEN-NEXT:    CLAUSE(TDLC_##Name, #Name, /* Implicit */ false) \
// GEN-NEXT:    CLAUSE_CLASS(TDLC_##Name, #Name, Class)
// GEN-NEXT:  #define __CLAUSE_NO_CLASS(Name)                    \
// GEN-NEXT:    CLAUSE(TDLC_##Name, #Name, /* Implicit */ false) \
// GEN-NEXT:    CLAUSE_NO_CLASS(TDLC_##Name, #Name)
// GEN-NEXT:  #define __IMPLICIT_CLAUSE_CLASS(Name, Str, Class)  \
// GEN-NEXT:    CLAUSE(TDLC_##Name, Str, /* Implicit */ true)    \
// GEN-NEXT:  CLAUSE_CLASS(TDLC_##Name, Str, Class)
// GEN-NEXT:  #define __IMPLICIT_CLAUSE_NO_CLASS(Name, Str)      \
// GEN-NEXT:    CLAUSE(TDLC_##Name, Str, /* Implicit */ true)    \
// GEN-NEXT:  CLAUSE_NO_CLASS(TDLC_##Name, Str)
// GEN-EMPTY:
// GEN-NEXT:  __IMPLICIT_CLAUSE_NO_CLASS(clausea, "clausea")
// GEN-NEXT:  __CLAUSE_NO_CLASS(clauseb)
// GEN-NEXT:  __CLAUSE(clausec, ClauseC)
// GEN-NEXT:  __IMPLICIT_CLAUSE_CLASS(claused, "claused", ClauseD)
// GEN-EMPTY:
// GEN-NEXT:  #undef __IMPLICIT_CLAUSE_NO_CLASS
// GEN-NEXT:  #undef __IMPLICIT_CLAUSE_CLASS
// GEN-NEXT:  #undef __CLAUSE
// GEN-NEXT:  #undef CLAUSE_NO_CLASS
// GEN-NEXT:  #undef CLAUSE_CLASS
// GEN-NEXT:  #undef CLAUSE
// GEN-EMPTY:
// GEN-NEXT:  #endif // GEN_CLANG_CLAUSE_CLASS


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
// GEN-NEXT:      .Case("clausea",TDLC_clauseb)
// GEN-NEXT:      .Case("clauseb",TDLC_clauseb)
// GEN-NEXT:      .Case("clausec",TDLC_clausec)
// GEN-NEXT:      .Case("claused",TDLC_clauseb)
// GEN-NEXT:      .Default(TDLC_clauseb);
// GEN-NEXT:  }
// GEN-EMPTY:
// GEN-NEXT:  llvm::StringRef llvm::tdl::getTdlClauseName(Clause Kind) {
// GEN-NEXT:    switch (Kind) {
// GEN-NEXT:      case TDLC_clausea:
// GEN-NEXT:        return "clausea";
// GEN-NEXT:      case TDLC_clauseb:
// GEN-NEXT:        return "clauseb";
// GEN-NEXT:      case TDLC_clausec:
// GEN-NEXT:        return "clausec";
// GEN-NEXT:      case TDLC_claused:
// GEN-NEXT:        return "claused";
// GEN-NEXT:    }
// GEN-NEXT:    llvm_unreachable("Invalid Tdl Clause kind");
// GEN-NEXT:  }
// GEN-EMPTY:
// GEN-NEXT:  bool llvm::tdl::isAllowedClauseForDirective(Directive D, Clause C, unsigned Version) {
// GEN-NEXT:    assert(unsigned(D) <= llvm::tdl::Directive_enumSize);
// GEN-NEXT:    assert(unsigned(C) <= llvm::tdl::Clause_enumSize);
// GEN-NEXT:    switch (D) {
// GEN-NEXT:      case TDLD_dira:
// GEN-NEXT:        switch (C) {
// GEN-NEXT:          case TDLC_clausea:
// GEN-NEXT:            return 2 <= Version && 4 >= Version;
// GEN-NEXT:          case TDLC_clauseb:
// GEN-NEXT:            return 2 <= Version && 2147483647 >= Version;
// GEN-NEXT:          default:
// GEN-NEXT:            return false;
// GEN-NEXT:        }
// GEN-NEXT:        break;
// GEN-NEXT:    }
// GEN-NEXT:    llvm_unreachable("Invalid Tdl Directive kind");
// GEN-NEXT:  }
// GEN-EMPTY:
// GEN-NEXT:  #endif // GEN_DIRECTIVES_IMPL
