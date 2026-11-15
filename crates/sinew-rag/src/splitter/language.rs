/// Languages with code-aware separator presets for
/// [`RecursiveCharacterTextSplitter::for_language`](super::RecursiveCharacterTextSplitter::for_language).
///
/// Separators are tried in order, so top-level definitions (functions,
/// classes, items) are preferred split points, then blank lines, lines,
/// words and finally characters.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum Language {
    Rust,
    Python,
    JavaScript,
    TypeScript,
    Go,
    Java,
    Cpp,
    Markdown,
    Html,
}

const TAIL: [&str; 4] = ["\n\n", "\n", " ", ""];

macro_rules! seps {
    ($($s:expr),* $(,)?) => {{
        const S: &[&str] = &[$($s,)* TAIL[0], TAIL[1], TAIL[2], TAIL[3]];
        S
    }};
}

impl Language {
    pub fn separators(self) -> &'static [&'static str] {
        match self {
            Language::Rust => seps![
                "\nfn ",
                "\npub fn ",
                "\nimpl",
                "\npub struct ",
                "\nstruct ",
                "\npub enum ",
                "\nenum ",
                "\npub trait ",
                "\ntrait ",
                "\nmod ",
                "\nconst ",
                "\nstatic ",
                "\n    fn ",
                "\n    pub fn ",
                "\n    let ",
                "\n    if ",
                "\n    for ",
                "\n    while ",
                "\n    loop ",
                "\n    match ",
            ],
            Language::Python => seps!["\nclass ", "\ndef ", "\n\tdef ", "\n    def "],
            Language::JavaScript => seps![
                "\nfunction ",
                "\nconst ",
                "\nlet ",
                "\nvar ",
                "\nclass ",
                "\nif ",
                "\nfor ",
                "\nwhile ",
                "\nswitch ",
                "\ncase ",
                "\ndefault ",
            ],
            Language::TypeScript => seps![
                "\nenum ",
                "\ninterface ",
                "\nnamespace ",
                "\ntype ",
                "\nclass ",
                "\nfunction ",
                "\nconst ",
                "\nlet ",
                "\nvar ",
                "\nif ",
                "\nfor ",
                "\nwhile ",
                "\nswitch ",
                "\ncase ",
                "\ndefault ",
            ],
            Language::Go => seps![
                "\nfunc ",
                "\nvar ",
                "\nconst ",
                "\ntype ",
                "\nif ",
                "\nfor ",
                "\nswitch ",
                "\ncase ",
            ],
            Language::Java => seps![
                "\nclass ",
                "\npublic ",
                "\nprotected ",
                "\nprivate ",
                "\nstatic ",
                "\nif ",
                "\nfor ",
                "\nwhile ",
                "\nswitch ",
                "\ncase ",
            ],
            Language::Cpp => seps![
                "\nclass ",
                "\nvoid ",
                "\nint ",
                "\nfloat ",
                "\ndouble ",
                "\nif ",
                "\nfor ",
                "\nwhile ",
                "\nswitch ",
                "\ncase ",
            ],
            Language::Markdown => seps![
                "\n# ",
                "\n## ",
                "\n### ",
                "\n#### ",
                "\n##### ",
                "\n###### ",
                "```\n",
                "\n***\n",
                "\n---\n",
                "\n___\n",
            ],
            Language::Html => seps![
                "<body", "<div", "<p", "<br", "<li", "<h1", "<h2", "<h3", "<h4", "<h5", "<h6",
                "<span", "<table", "<tr", "<td", "<th", "<ul", "<ol", "<header", "<footer", "<nav",
                "<head", "<style", "<script", "<meta", "<title",
            ],
        }
    }

    /// Guesses the language from a file extension (`"rs"`, `"py"`, ...).
    pub fn from_extension(ext: &str) -> Option<Self> {
        Some(match ext.to_ascii_lowercase().as_str() {
            "rs" => Language::Rust,
            "py" | "pyi" => Language::Python,
            "js" | "mjs" | "cjs" | "jsx" => Language::JavaScript,
            "ts" | "tsx" | "mts" => Language::TypeScript,
            "go" => Language::Go,
            "java" => Language::Java,
            "c" | "cc" | "cpp" | "cxx" | "h" | "hpp" => Language::Cpp,
            "md" | "markdown" | "mdx" => Language::Markdown,
            "html" | "htm" => Language::Html,
            _ => return None,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn presets_end_with_generic_separators() {
        for l in [Language::Rust, Language::Python, Language::Markdown] {
            assert_eq!(&l.separators()[l.separators().len() - 4..], &TAIL);
        }
        assert_eq!(Language::from_extension("RS"), Some(Language::Rust));
        assert_eq!(Language::from_extension("zz"), None);
    }
}
