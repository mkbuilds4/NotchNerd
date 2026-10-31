//
//  GuardModelConverter.swift
//  NotchNerd
//
//  Turns InsightFace's w600k_mbf.onnx (MobileFaceNet, ArcFace loss) into a Core ML
//  NeuralNetwork spec on the user's Mac, so the weights never ship inside NotchNerd.
//  Covers exactly the ops that model uses (Conv, PRelu, Add, Flatten, Gemm, BatchNormalization)
//  and refuses anything else rather than guessing.
//

import Foundation

enum GuardModelConverter {
    struct ConversionError: LocalizedError {
        let message: String
        var errorDescription: String? { message }
    }

    /// ONNX model bytes in, `.mlmodel` (protobuf spec) bytes out.
    /// The input is a 112x112 RGB image named "face", normalized to [-1, 1] like InsightFace's
    /// own pipeline ((x - 127.5) / 127.5); the output is the raw 512-d embedding named "embedding".
    static func convert(onnx data: Data) throws -> Data {
        let graph = try ONNXGraph(model: data)
        guard let input = graph.inputs.first(where: { graph.initializers[$0] == nil }) else {
            throw ConversionError(message: "model has no image input")
        }
        guard let output = graph.outputs.first else { throw ConversionError(message: "model has no output") }

        func name(_ t: String) -> String { t == input ? "face" : t == output ? "embedding" : "t_" + t }
        func weight(_ t: String) throws -> ONNXTensor {
            guard let w = graph.initializers[t] else { throw ConversionError(message: "missing weight \(t)") }
            return w
        }

        var layers: [PBWriter] = []
        for (i, node) in graph.nodes.enumerated() {
            var layer = PBWriter() // the op's params; name/inputs/outputs go in front below
            var inputs: [String] = []
            switch node.op {
            case "Conv":
                let w = try weight(node.inputs[1])
                guard w.dims.count == 4 else { throw ConversionError(message: "Conv \(i): weight rank \(w.dims.count)") }
                let groups = UInt64(node.int("group") ?? 1)
                let kernel = node.ints("kernel_shape") ?? [w.dims[2], w.dims[3]]
                let strides = node.ints("strides") ?? [1, 1]
                let dilations = node.ints("dilations") ?? [1, 1]
                let pads = node.ints("pads") ?? [0, 0, 0, 0] // [top, left, bottom, right]
                guard kernel.count == 2, strides.count == 2, dilations.count == 2, pads.count == 4 else {
                    throw ConversionError(message: "Conv \(i): only 2-D convolutions are supported")
                }
                inputs = [node.inputs[0]]
                layer.message(100) { conv in
                    conv.uint(1, UInt64(w.dims[0]))
                    conv.uint(2, UInt64(w.dims[1]))
                    conv.uint(10, groups)
                    conv.packed(20, kernel.map(UInt64.init))
                    conv.packed(30, strides.map(UInt64.init))
                    conv.packed(40, dilations.map(UInt64.init))
                    conv.message(50) { valid in
                        valid.message(1) { amounts in
                            amounts.message(10) { $0.uint(1, UInt64(pads[0])); $0.uint(2, UInt64(pads[2])) } // height
                            amounts.message(10) { $0.uint(1, UInt64(pads[1])); $0.uint(2, UInt64(pads[3])) } // width
                        }
                    }
                    if node.inputs.count > 2 {
                        conv.bool(70, true)
                        conv.message(90) { $0.floats(1, w.floats) } // OIHW, same layout Core ML wants
                        let b = try? weight(node.inputs[2])
                        conv.message(91) { $0.floats(1, b?.floats ?? []) }
                    } else {
                        conv.message(90) { $0.floats(1, w.floats) }
                    }
                }
            case "PRelu":
                let slope = try weight(node.inputs[1])
                inputs = [node.inputs[0]]
                layer.message(130) { act in act.message(25) { prelu in prelu.message(1) { $0.floats(1, slope.floats) } } }
            case "Add":
                guard node.inputs.allSatisfy({ graph.initializers[$0] == nil }) else {
                    throw ConversionError(message: "Add \(i): constant operands are not supported")
                }
                inputs = node.inputs
                layer.message(230) { _ in }
            case "Flatten":
                guard (node.int("axis") ?? 1) == 1 else { throw ConversionError(message: "Flatten \(i): axis must be 1") }
                inputs = [node.inputs[0]]
                layer.message(301) { _ in } // CHANNEL_FIRST, the ONNX (NCHW) order
            case "Gemm":
                guard (node.float("alpha") ?? 1) == 1, (node.float("beta") ?? 1) == 1, (node.int("transA") ?? 0) == 0 else {
                    throw ConversionError(message: "Gemm \(i): only alpha = beta = 1, transA = 0 is supported")
                }
                let w = try weight(node.inputs[1])
                let transB = (node.int("transB") ?? 0) == 1
                // Core ML wants [out, in]; ONNX stores [out, in] when transB = 1, [in, out] otherwise.
                let (outCh, inCh) = transB ? (w.dims[0], w.dims[1]) : (w.dims[1], w.dims[0])
                let weights = transB ? w.floats : transpose(w.floats, rows: w.dims[0], cols: w.dims[1])
                let bias = node.inputs.count > 2 ? try weight(node.inputs[2]).floats : nil
                inputs = [node.inputs[0]]
                layer.message(140) { fc in
                    fc.uint(1, UInt64(inCh))
                    fc.uint(2, UInt64(outCh))
                    if let bias {
                        fc.bool(10, true)
                        fc.message(20) { $0.floats(1, weights) }
                        fc.message(21) { $0.floats(1, bias) }
                    } else {
                        fc.message(20) { $0.floats(1, weights) }
                    }
                }
            case "BatchNormalization":
                let gamma = try weight(node.inputs[1]), beta = try weight(node.inputs[2])
                let mean = try weight(node.inputs[3]), variance = try weight(node.inputs[4])
                inputs = [node.inputs[0]]
                layer.message(160) { bn in
                    bn.uint(1, UInt64(gamma.floats.count))
                    bn.float(10, node.float("epsilon") ?? 1e-5)
                    bn.message(15) { $0.floats(1, gamma.floats) }
                    bn.message(16) { $0.floats(1, beta.floats) }
                    bn.message(17) { $0.floats(1, mean.floats) }
                    bn.message(18) { $0.floats(1, variance.floats) }
                }
            default:
                throw ConversionError(message: "unsupported op \(node.op)")
            }
            var header = PBWriter()
            header.string(1, "\(node.op.lowercased())_\(i)")
            inputs.forEach { header.string(2, name($0)) }
            node.outputs.forEach { header.string(3, name($0)) }
            header.data.append(layer.data)
            layers.append(header)
        }

        var model = PBWriter()
        model.int(1, 4) // specificationVersion 4 (macOS 10.15+)
        model.message(2) { desc in
            desc.message(1) { feature in
                feature.string(1, "face")
                feature.string(2, "Aligned 112x112 face crop")
                feature.message(3) { $0.message(4) { img in img.int(1, 112); img.int(2, 112); img.int(3, 20) } } // RGB
            }
            desc.message(10) { feature in
                feature.string(1, "embedding")
                feature.string(2, "512-d face embedding (not normalized)")
                feature.message(3) { $0.message(5) { arr in arr.packed(1, [Int64(512)]); arr.int(2, 65568) } } // FLOAT32
            }
            desc.message(100) { meta in
                meta.string(1, "InsightFace w600k_mbf (MobileFaceNet), converted on this Mac by NotchNerd Guard")
                meta.string(3, "InsightFace (deepinsight/insightface)")
                meta.string(4, "InsightFace pretrained models: non-commercial research use only")
            }
        }
        model.message(500) { nn in
            layers.forEach { l in nn.message(1) { $0.data.append(l.data) } }
            nn.message(2) { pre in
                pre.string(1, "face")
                pre.message(10) { s in
                    s.float(10, 1 / 127.5)
                    s.float(20, -1); s.float(21, -1); s.float(22, -1)
                }
            }
        }
        return model.data
    }

    private static func transpose(_ a: [Float], rows: Int, cols: Int) -> [Float] {
        var out = [Float](repeating: 0, count: a.count)
        for r in 0..<rows { for c in 0..<cols { out[c * rows + r] = a[r * cols + c] } }
        return out
    }
}

// MARK: - Minimal protobuf writer

struct PBWriter {
    var data = Data()

    mutating func varint(_ v: UInt64) {
        var v = v
        while v >= 0x80 { data.append(UInt8(v & 0x7F) | 0x80); v >>= 7 }
        data.append(UInt8(v))
    }
    mutating func key(_ field: Int, _ wire: UInt64) { varint(UInt64(field) << 3 | wire) }
    mutating func uint(_ field: Int, _ v: UInt64) { key(field, 0); varint(v) }
    mutating func int(_ field: Int, _ v: Int64) { key(field, 0); varint(UInt64(bitPattern: v)) }
    mutating func bool(_ field: Int, _ v: Bool) { uint(field, v ? 1 : 0) }
    mutating func float(_ field: Int, _ v: Float) {
        key(field, 5)
        withUnsafeBytes(of: v.bitPattern.littleEndian) { data.append(contentsOf: $0) }
    }
    mutating func bytes(_ field: Int, _ d: Data) { key(field, 2); varint(UInt64(d.count)); data.append(d) }
    mutating func string(_ field: Int, _ s: String) { bytes(field, Data(s.utf8)) }
    mutating func message(_ field: Int, _ build: (inout PBWriter) throws -> Void) rethrows {
        var sub = PBWriter()
        try build(&sub)
        bytes(field, sub.data)
    }
    mutating func packed(_ field: Int, _ values: [UInt64]) {
        var sub = PBWriter()
        values.forEach { sub.varint($0) }
        bytes(field, sub.data)
    }
    mutating func packed(_ field: Int, _ values: [Int64]) { packed(field, values.map { UInt64(bitPattern: $0) }) }
    mutating func floats(_ field: Int, _ values: [Float]) {
        var d = Data(capacity: values.count * 4)
        values.forEach { v in withUnsafeBytes(of: v.bitPattern.littleEndian) { d.append(contentsOf: $0) } }
        bytes(field, d)
    }
}

// MARK: - Minimal protobuf reader + the slice of the ONNX schema we need

struct PBReader {
    enum Value { case varint(UInt64), fixed64(UInt64), bytes(Data), fixed32(UInt32) }
    struct Malformed: Error {}

    let data: Data
    private(set) var offset: Int

    init(_ data: Data) { self.data = data; offset = data.startIndex }
    var atEnd: Bool { offset >= data.endIndex }

    mutating func varint() throws -> UInt64 {
        var result: UInt64 = 0, shift: UInt64 = 0
        while true {
            guard offset < data.endIndex, shift < 64 else { throw Malformed() }
            let b = data[offset]; offset += 1
            result |= UInt64(b & 0x7F) << shift
            if b < 0x80 { return result }
            shift += 7
        }
    }

    mutating func next() throws -> (Int, Value)? {
        guard !atEnd else { return nil }
        let k = try varint()
        let field = Int(k >> 3)
        switch k & 7 {
        case 0: return (field, .varint(try varint()))
        case 1:
            guard offset + 8 <= data.endIndex else { throw Malformed() }
            let v = data[offset..<offset + 8].withUnsafeBytes { $0.loadUnaligned(as: UInt64.self) }
            offset += 8
            return (field, .fixed64(UInt64(littleEndian: v)))
        case 2:
            let n = Int(try varint())
            guard n >= 0, offset + n <= data.endIndex else { throw Malformed() }
            let d = data[offset..<offset + n]
            offset += n
            return (field, .bytes(Data(d)))
        case 5:
            guard offset + 4 <= data.endIndex else { throw Malformed() }
            let v = data[offset..<offset + 4].withUnsafeBytes { $0.loadUnaligned(as: UInt32.self) }
            offset += 4
            return (field, .fixed32(UInt32(littleEndian: v)))
        default: throw Malformed()
        }
    }

    static func each(_ data: Data, _ body: (Int, Value) throws -> Void) throws {
        var r = PBReader(data)
        while let (f, v) = try r.next() { try body(f, v) }
    }

    /// Repeated int64 fields arrive either packed (one bytes blob) or one varint per entry.
    static func ints(_ v: Value) throws -> [Int64] {
        switch v {
        case .varint(let x): return [Int64(bitPattern: x)]
        case .bytes(let d):
            var r = PBReader(d), out: [Int64] = []
            while !r.atEnd { out.append(Int64(bitPattern: try r.varint())) }
            return out
        default: throw Malformed()
        }
    }

    static func string(_ v: Value) -> String? {
        if case .bytes(let d) = v { return String(data: d, encoding: .utf8) }
        return nil
    }
}

struct ONNXTensor {
    var dims: [Int] = []
    var floats: [Float] = []
}

struct ONNXNode {
    var op = ""
    var inputs: [String] = []
    var outputs: [String] = []
    var attributes: [String: PBReader.Value] = [:]
    var intLists: [String: [Int]] = [:]

    func int(_ name: String) -> Int? {
        if case .varint(let v)? = attributes[name] { return Int(Int64(bitPattern: v)) }
        return nil
    }
    func float(_ name: String) -> Float? {
        if case .fixed32(let v)? = attributes[name] { return Float(bitPattern: v) }
        return nil
    }
    func ints(_ name: String) -> [Int]? { intLists[name] }
}

struct ONNXGraph {
    var nodes: [ONNXNode] = []
    var initializers: [String: ONNXTensor] = [:]
    var inputs: [String] = []
    var outputs: [String] = []

    init(model: Data) throws {
        var graphData: Data?
        try PBReader.each(model) { f, v in if f == 7, case .bytes(let d) = v { graphData = d } } // ModelProto.graph
        guard let graphData else { throw GuardModelConverter.ConversionError(message: "not an ONNX model") }
        try PBReader.each(graphData) { f, v in
            guard case .bytes(let d) = v else { return }
            switch f {
            case 1: nodes.append(try Self.node(d))
            case 5: let (n, t) = try Self.tensor(d); initializers[n] = t
            case 11: if let n = try Self.valueInfoName(d) { inputs.append(n) }
            case 12: if let n = try Self.valueInfoName(d) { outputs.append(n) }
            default: break
            }
        }
    }

    private static func valueInfoName(_ d: Data) throws -> String? {
        var name: String?
        try PBReader.each(d) { f, v in if f == 1 { name = PBReader.string(v) } }
        return name
    }

    private static func node(_ d: Data) throws -> ONNXNode {
        var n = ONNXNode()
        try PBReader.each(d) { f, v in
            switch f {
            case 1: n.inputs.append(PBReader.string(v) ?? "")
            case 2: n.outputs.append(PBReader.string(v) ?? "")
            case 4: n.op = PBReader.string(v) ?? ""
            case 5:
                guard case .bytes(let a) = v else { return }
                var name = "", scalar: PBReader.Value?, list: [Int] = []
                try PBReader.each(a) { af, av in
                    switch af {
                    case 1: name = PBReader.string(av) ?? ""
                    case 2, 3: scalar = av // f (float) / i (int)
                    case 8: list += try PBReader.ints(av).map { Int($0) } // ints
                    default: break
                    }
                }
                if let scalar { n.attributes[name] = scalar }
                if !list.isEmpty { n.intLists[name] = list }
            default: break
            }
        }
        return n
    }

    private static func tensor(_ d: Data) throws -> (String, ONNXTensor) {
        var name = "", t = ONNXTensor(), type: UInt64 = 1, raw: Data?
        try PBReader.each(d) { f, v in
            switch f {
            case 1: t.dims += try PBReader.ints(v).map { Int($0) }
            case 2: if case .varint(let x) = v { type = x }
            case 4: // float_data, packed or not
                switch v {
                case .fixed32(let x): t.floats.append(Float(bitPattern: x))
                case .bytes(let b): t.floats += floats(b)
                default: break
                }
            case 8: name = PBReader.string(v) ?? ""
            case 9: if case .bytes(let b) = v { raw = b }
            default: break
            }
        }
        guard type == 1 else { throw GuardModelConverter.ConversionError(message: "tensor \(name): only float32 weights are supported") }
        if let raw { t.floats = floats(raw) }
        let expected = t.dims.reduce(1, *)
        guard t.floats.count == expected else {
            throw GuardModelConverter.ConversionError(message: "tensor \(name): \(t.floats.count) values for shape \(t.dims)")
        }
        return (name, t)
    }

    private static func floats(_ b: Data) -> [Float] {
        stride(from: 0, to: b.count - 3, by: 4).map { i in
            Float(bitPattern: UInt32(littleEndian: b.withUnsafeBytes { $0.loadUnaligned(fromByteOffset: i, as: UInt32.self) }))
        }
    }
}
