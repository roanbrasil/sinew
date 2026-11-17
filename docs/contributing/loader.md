# Adding a document loader

Loaders live in `crates/sinew-rag/src/loaders/`, one module per format. `csv.rs` is a compact reference with
builder options; `pdf.rs` shows a loader behind a feature flag.

## The trait

From `crates/sinew-rag/src/loaders/mod.rs`:

```rust
/// Produces documents.
pub trait Loader: Send + Sync {
    fn load(&self) -> Result<Vec<Document>>;
}
```

Any `Fn() -> Result<Vec<Document>> + Send + Sync` closure is also a `Loader`, so one-off sources do not need a
type. Add a loader to the crate when the format is common enough that others will need it.

Loaders are **synchronous**: file parsing runs once at indexing time, and callers in async servers wrap them in
`tokio::task::spawn_blocking`. A loader that has to fetch over the network (a web page, a cloud bucket) is better
written as an async function in your application, or proposed in an issue first.

Conventions every loader follows:

- Every document gets a `source` metadata field with its path: start from
  `source_metadata(&self.path)` and add fields with `Document::with_metadata` / `with_metadata_map`.
- Read files with `read_to_string(&self.path)` (invalid UTF-8 is replaced, not an error).
- Errors carry the path: `Error::io(path, e)` for I/O and `Error::Load { path, message }` for parse errors.
- Formats with natural units (rows, pages, records) produce one document per unit and record its position in
  metadata (`row`, `page`, `seq_num`), with an option to load the whole file as one document where that makes
  sense (`PdfLoader::single`).

## Steps

1. **Module.** Create `crates/sinew-rag/src/loaders/acme.rs`:

   ```rust
   use std::path::PathBuf;

   use super::{read_to_string, source_metadata, Loader};
   use crate::{Document, Error, Result};

   /// Loads an Acme file, one document per section. Every document gets
   /// `source` and `section` (0-based) metadata.
   #[derive(Debug, Clone)]
   pub struct AcmeLoader {
       path: PathBuf,
   }

   impl AcmeLoader {
       pub fn new(path: impl Into<PathBuf>) -> Self {
           Self { path: path.into() }
       }
   }

   impl Loader for AcmeLoader {
       fn load(&self) -> Result<Vec<Document>> {
           let text = read_to_string(&self.path)?;
           let sections = parse(&text).map_err(|message| Error::Load {
               path: self.path.clone(),
               message,
           })?;
           Ok(sections
               .into_iter()
               .enumerate()
               .map(|(i, s)| {
                   Document::new(s)
                       .with_metadata_map(source_metadata(&self.path))
                       .with_metadata("section", i)
               })
               .collect())
       }
   }
   ```

2. **Register it** in `crates/sinew-rag/src/loaders/mod.rs` (`mod acme;` and `pub use acme::AcmeLoader;`) and
   re-export it from `crates/sinew-rag/src/lib.rs`.

3. **Directory loader.** Add the file extension(s) to the `match ext.as_str()` in `DirectoryLoader::load_file`
   (`crates/sinew-rag/src/loaders/directory.rs`) so `DirectoryLoader` picks the loader automatically, and to the
   extension list in its doc comment.

4. **Feature flag.** Only needed for a new dependency. Parsers that pull in large crates go behind a feature, as
   `pdf` does:

   ```toml
   # crates/sinew-rag/Cargo.toml
   [features]
   # `AcmeLoader`: Acme files via the `acme-parser` crate.
   acme = ["dep:acme-parser"]
   ```

   Gate the module, the `pub use` lines and the `DirectoryLoader` arm with `#[cfg(feature = "acme")]`. Forward
   the feature from `crates/sinew/Cargo.toml` (`acme-loader = ["rag", "sinew-rag/acme"]`) and add it to `full`
   when it has no system dependency. A loader written with the standard library or existing dependencies needs
   no feature.

5. **Tests** in a `#[cfg(test)] mod tests` in the module. Write a small fixture file to a `tempfile::tempdir()`
   (or commit a tiny one under `crates/sinew-rag/tests/fixtures/` for binary formats) and check content,
   metadata (`source` and the position field), options, and that a malformed file returns `Error::Load` with the
   path. `loads_rows` in `csv.rs` is the pattern. Add a case to the `DirectoryLoader` tests if you registered an
   extension.

## Docs to update

- `docs/rag.md`: a row in the loader table (constructor, input, metadata it adds) and the extension list under
  it.
- The module docs of `crates/sinew-rag/src/loaders/mod.rs` and the loader list in `crates/sinew-rag/src/lib.rs`.
- `README.md`: the loader list in the `sinew-rag` row.

## CHANGELOG

```markdown
- `AcmeLoader` in `sinew-rag`: Acme files, one document per section with `source` and `section` metadata; `DirectoryLoader` picks it for `.acme` (#123).
```
