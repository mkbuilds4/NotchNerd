//
//  GuardModelInstaller.swift
//  NotchNerd
//
//  First-run face model install. InsightFace's pretrained models are licensed for non-commercial
//  research use only, so the weights are never shipped: this fetches w600k_mbf.onnx from InsightFace's
//  own GitHub release, checks its SHA-256, and converts it to Core ML on this Mac.
//
//  Only the one 13 MB zip entry is downloaded (HTTP Range requests against the 128 MB buffalo_s.zip);
//  if the server won't serve ranges it falls back to the whole zip.
//

import Compression
import CoreML
import CryptoKit
import Foundation

enum GuardModelInstaller {
    static let releaseZip = URL(string: "https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_s.zip")!
    static let entryName = "w600k_mbf.onnx"
    static let entrySHA256 = "9cc6e4a75f0e2bf0b1aed94578f144d15175f357bdc05e815e5c4a02b319eb4f"
    static let licenseURL = URL(string: "https://github.com/deepinsight/insightface/tree/master/python-package#model-zoo")!

    struct InstallError: LocalizedError {
        let message: String
        var errorDescription: String? { message }
    }

    /// Downloads, verifies, converts and compiles the model to `destination` (a `.mlmodelc` path).
    /// `progress` gets 0...1 for the download; conversion takes well under a second after that.
    static func install(to destination: URL, progress: @escaping @Sendable (Double) -> Void = { _ in }) async throws {
        let onnx = try await fetchEntry(progress: progress)
        let digest = SHA256.hash(data: onnx).map { String(format: "%02x", $0) }.joined()
        guard digest == entrySHA256 else {
            throw InstallError(message: "The downloaded model doesn't match the expected checksum")
        }
        let spec = try GuardModelConverter.convert(onnx: onnx)
        let tmp = FileManager.default.temporaryDirectory.appendingPathComponent("FaceEmbedder-\(UUID().uuidString).mlmodel")
        try spec.write(to: tmp)
        defer { try? FileManager.default.removeItem(at: tmp) }
        let compiled = try await MLModel.compileModel(at: tmp)
        // Load it once so a broken compile fails here, not at the first check.
        _ = try MLModel(contentsOf: compiled)
        let fm = FileManager.default
        try fm.createDirectory(at: destination.deletingLastPathComponent(), withIntermediateDirectories: true)
        if fm.fileExists(atPath: destination.path) { try fm.removeItem(at: destination) }
        try fm.moveItem(at: compiled, to: destination)
    }

    // MARK: - Zip entry over HTTP ranges

    private struct Entry { let method: UInt16; let compressedSize: Int; let size: Int; let localHeader: Int }

    private static func fetchEntry(progress: @escaping @Sendable (Double) -> Void) async throws -> Data {
        // The end-of-central-directory record sits in the last 22 bytes (plus any comment).
        guard let (tail, total) = try await range(suffix: 64 * 1024) else { return try await fetchWholeZip(progress: progress) }
        guard let eocd = lastIndex(of: [0x50, 0x4B, 0x05, 0x06], in: tail) else { throw InstallError(message: "Not a zip archive") }
        let cdSize = Int(tail.u32(eocd + 12)), cdOffset = Int(tail.u32(eocd + 16))
        let tailStart = total - tail.count
        let directory = cdOffset >= tailStart
            ? tail.subdata(in: (cdOffset - tailStart)..<(cdOffset - tailStart + cdSize))
            : try await range(cdOffset, cdOffset + cdSize - 1)
        let entry = try find(entryName, in: directory)

        let header = try await range(entry.localHeader, entry.localHeader + 29)
        guard header.u32(0) == 0x0403_4B50 else { throw InstallError(message: "Bad zip entry header") }
        let dataStart = entry.localHeader + 30 + Int(header.u16(26)) + Int(header.u16(28))
        let body = try await range(dataStart, dataStart + entry.compressedSize - 1, progress: progress)
        return try decode(body, entry)
    }

    private static func find(_ name: String, in cd: Data) throws -> Entry {
        var i = 0
        while i + 46 <= cd.count, cd.u32(i) == 0x0201_4B50 {
            let nameLen = Int(cd.u16(i + 28)), extraLen = Int(cd.u16(i + 30)), commentLen = Int(cd.u16(i + 32))
            let entryName = String(decoding: cd.subdata(in: (i + 46)..<(i + 46 + nameLen)), as: UTF8.self)
            if entryName == name {
                return Entry(method: cd.u16(i + 10), compressedSize: Int(cd.u32(i + 20)), size: Int(cd.u32(i + 24)),
                             localHeader: Int(cd.u32(i + 42)))
            }
            i += 46 + nameLen + extraLen + commentLen
        }
        throw InstallError(message: "\(name) isn't in the InsightFace archive")
    }

    private static func decode(_ body: Data, _ e: Entry) throws -> Data {
        switch e.method {
        case 0: return body
        case 8: // raw DEFLATE, which is what COMPRESSION_ZLIB decodes
            var out = Data(count: e.size)
            let n = out.withUnsafeMutableBytes { dst in
                body.withUnsafeBytes { src in
                    compression_decode_buffer(dst.bindMemory(to: UInt8.self).baseAddress!, e.size,
                                              src.bindMemory(to: UInt8.self).baseAddress!, body.count, nil, COMPRESSION_ZLIB)
                }
            }
            guard n == e.size else { throw InstallError(message: "Couldn't unpack the model") }
            return out
        default: throw InstallError(message: "Unsupported zip compression (\(e.method))")
        }
    }

    /// The last `n` bytes plus the file's total size, or nil if the server ignores Range.
    private static func range(suffix n: Int) async throws -> (Data, Int)? {
        var req = URLRequest(url: releaseZip)
        req.setValue("bytes=-\(n)", forHTTPHeaderField: "Range")
        let (data, resp) = try await URLSession.shared.data(for: req)
        guard let http = resp as? HTTPURLResponse, http.statusCode == 206,
              let total = (http.value(forHTTPHeaderField: "Content-Range")?.split(separator: "/").last).flatMap({ Int($0) })
        else { return nil }
        return (data, total)
    }

    private static func range(_ from: Int, _ to: Int, progress: (@Sendable (Double) -> Void)? = nil) async throws -> Data {
        var req = URLRequest(url: releaseZip)
        req.setValue("bytes=\(from)-\(to)", forHTTPHeaderField: "Range")
        let (file, resp) = try await URLSession.shared.download(for: req, delegate: progress.map(ProgressRelay.init))
        defer { try? FileManager.default.removeItem(at: file) }
        guard (resp as? HTTPURLResponse)?.statusCode == 206 else { throw InstallError(message: "Download failed") }
        let data = try Data(contentsOf: file)
        progress?(1)
        guard data.count == to - from + 1 else { throw InstallError(message: "Download was cut short") }
        return data
    }

    private final class ProgressRelay: NSObject, URLSessionDownloadDelegate, @unchecked Sendable {
        let report: @Sendable (Double) -> Void
        init(_ report: @escaping @Sendable (Double) -> Void) { self.report = report }
        func urlSession(_ s: URLSession, downloadTask: URLSessionDownloadTask, didWriteData _: Int64,
                        totalBytesWritten written: Int64, totalBytesExpectedToWrite expected: Int64) {
            if expected > 0 { report(Double(written) / Double(expected)) }
        }
        func urlSession(_ s: URLSession, downloadTask: URLSessionDownloadTask, didFinishDownloadingTo location: URL) {}
    }

    /// Fallback: the whole archive, then `unzip -p` for the one entry.
    private static func fetchWholeZip(progress: @escaping @Sendable (Double) -> Void) async throws -> Data {
        let (file, resp) = try await URLSession.shared.download(from: releaseZip)
        defer { try? FileManager.default.removeItem(at: file) }
        guard (resp as? HTTPURLResponse)?.statusCode == 200 else { throw InstallError(message: "Download failed") }
        progress(1)
        let p = Process(), out = Pipe()
        p.executableURL = URL(fileURLWithPath: "/usr/bin/unzip")
        p.arguments = ["-p", file.path, entryName]
        p.standardOutput = out
        try p.run()
        let data = out.fileHandleForReading.readDataToEndOfFile()
        p.waitUntilExit()
        guard p.terminationStatus == 0 else { throw InstallError(message: "Couldn't unpack the model") }
        return data
    }

    private static func lastIndex(of sig: [UInt8], in d: Data) -> Int? {
        guard d.count >= sig.count else { return nil }
        for i in stride(from: d.count - sig.count, through: 0, by: -1) where d[d.startIndex + i] == sig[0] {
            if (1..<sig.count).allSatisfy({ d[d.startIndex + i + $0] == sig[$0] }) { return i }
        }
        return nil
    }
}

private extension Data {
    func u16(_ i: Int) -> UInt16 { UInt16(self[startIndex + i]) | UInt16(self[startIndex + i + 1]) << 8 }
    func u32(_ i: Int) -> UInt32 { UInt32(u16(i)) | UInt32(u16(i + 2)) << 16 }
}
