//===- FIRTypes.td - FIR types -----------------------------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares the FIR dialect types.
//
//===----------------------------------------------------------------------===//

#ifndef FIR_DIALECT_FIR_TYPES
#define FIR_DIALECT_FIR_TYPES

//===----------------------------------------------------------------------===//
// FIR Types
//===----------------------------------------------------------------------===//

class FIR_Type<string name, string typeMnemonic> : TypeDef<fir_Dialect, name> {
  let mnemonic = typeMnemonic;
}

def BoxType : FIR_Type<"Box", "box"> {
  let summary = "A descriptor tuple (captures a reference to an entity and other information)";
  let description = [{
    The type of a Fortran descriptor. Descriptors are tuples of information that
    describe an entity being passed from a calling context. This information
    might include (but is not limited to) whether the entity is an array, its
    size, or what type it has.
  }];

  let parameters = (ins "mlir::Type":$eleTy, "mlir::AffineMapAttr":$map);

  let printer = [{
    $_printer << "box<";
    $_printer.printType(getEleTy());
    if (auto map = getLayoutMap()) {
      $_printer << ", ";
      $_printer.printAttribute(map);
    }
    $_printer << '>';
  }];

  // `box` `<` type (',' affine-map)? `>`
  let parser = [{
    mlir::Type ofTy;
    if ($_parser.parseLess() || $_parser.parseType(ofTy)) {
      $_parser.emitError(parser.getCurrentLocation(), "expected type parameter");
      return Type();
    }

    mlir::AffineMapAttr map;
    if (!$_parser.parseOptionalComma())
      if ($_parser.parseAttribute(map)) {
        $_parser.emitError($_parser.getCurrentLocation(), "expected affine map");
        return {};
      }
    if ($_parser.parseGreater()) {
      $_parser.emitError($_parser.getCurrentLocation(), "expected '>'");
      return {};
    }
    return get(ofTy, map);
  }];

  let extraClassDeclaration = [{
    mlir::Type getElementType() const { return getEleTy(); }
    mlir::AffineMapAttr getLayoutMap() const { return getMap(); }
    static BoxType get(mlir::Type eleTy, mlir::AffineMapAttr map = {}) {
      return get(eleTy, map);
    }
  }];

  let genAccessors = 1;
  let genVerifyInvariantsDecl = 1;
}

def ShapeType : FIR_Type<"Shape", "shape"> {
  let summary = "shape of a multidimensional array object";

  let description = [{
    Type of a vector of runtime values that define the shape of a
    multidimensional array object. The vector is the extents of each array
    dimension. The rank of a ShapeType must be at least 1.
  }];

  let parameters = (ins "unsigned":$rank);

  let printer = [{
    $_printer << "shape<" << getImpl()->rank << ">";
  }];

  let parser = [{
    if ($_parser.parseLess())
      return Type();
    int rank;
    if ($_parser.parseInteger(rank))
      return Type();
    if ($_parser.parseGreater())
      return Type();
    return get(context, rank);
  }];
}

def A_CharacterType : FIR_Type<"Character", "char"> {
  let summary = "FIR character type";

  let description = [{
    Model of the Fortran CHARACTER intrinsic type, including the KIND type
    parameter. The model optionally includes a LEN type parameter. A
    CharacterType is thus the type of both a single character value and a
    character with a LEN parameter.
  }];

  let parameters = (ins "KindTy":$FKind, "CharacterType::LenType":$len);

  let printer = [{
    $_printer << "char<" << getImpl()->FKind;
    auto len = getImpl()->len;
    if (len != fir::CharacterType::singleton()) {
      $_printer << ',';
      if (len == fir::CharacterType::unknownLen())
        $_printer << '?';
      else
        $_printer << len;
    }
    $_printer << '>';
  }];

  // `char` `<` kind [`,` `len`] `>`
  let parser = [{
    int kind = 0;
    if ($_parser.parseLess() || $_parser.parseInteger(kind)) {
      $_parser.emitError($_parser.getCurrentLocation(), "kind value expected");
      return Type();
    }
    CharacterType::LenType len = 1;
    if (mlir::succeeded($_parser.parseOptionalComma())) {
      if (mlir::succeeded($_parser.parseOptionalQuestion())) {
        len = fir::CharacterType::unknownLen();
      } else if (!mlir::succeeded($_parser.parseInteger(len))) {
        $_parser.emitError($_parser.getCurrentLocation(), "len value expected");
        return Type();
      }
    }
    if ($_parser.parseGreater())
      return Type();
    return get(context, kind, len);
  }];

  let extraClassDeclaration = [{
    using KindTy = unsigned;
    using LenType = std::int64_t;

    // Return unknown length CHARACTER type.
    static CharacterType getUnknownLen(mlir::MLIRContext *ctxt, KindTy kind) {
      return get(ctxt, kind, unknownLen());
    }

    // Return length 1 CHARACTER type.
    static CharacterType getSingleton(mlir::MLIRContext *ctxt, KindTy kind) {
      return get(ctxt, kind, singleton());
    }

    // CHARACTER is a singleton and has a LEN of 1.
    static constexpr LenType singleton() { return 1; }
    // CHARACTER has an unknown LEN property.
    static constexpr LenType unknownLen() { return -1; }
  }];
}

def BoxCharType : FIR_Type<"BoxChar", "boxchar"> {
  let summary = "CHARACTER type descriptor.";

  let description = [{
    The type of a pair that describes a CHARACTER variable. Specifically, a
    CHARACTER consists of a reference to a buffer (the string value) and a LEN
    type parameter (the runtime length of the buffer).
  }];

  let parameters = (ins "KindTy":$kind);

  let printer = [{
    $_printer << "boxchar<" << getImpl()->kind << ">";
  }];

  let parser = [{
    int kind = 0;
    if ($_parser.parseLess() || $_parser.parseInteger(kind) ||
      $_parser.parseGreater()) {
      $_parser.emitError($_parser.getCurrentLocation(), "kind value expected");
      return Type();
    }
    return get(context, kind);
  }];

  let genAccessors = 1;

  let extraClassDeclaration = [{
    using KindTy = unsigned;

    // a !fir.boxchar<k> always wraps a !fir.char<k, ?>
    CharacterType getElementType(mlir::MLIRContext *ctxt) const {
      return CharacterType::getUnknownLen(ctxt, getKind());
    }

    CharacterType getEleTy() const {
      return getElementType(getContext());
    }
  }];


}

#endif // FIR_DIALECT_FIR_TYPES
