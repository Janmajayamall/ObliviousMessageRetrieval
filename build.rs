fn main() -> std::io::Result<()> {
    prost_build::compile_protos(&["src/pvw/proto/pvw.proto"], &["src/pvw/proto"])?;
    Ok(())
}
