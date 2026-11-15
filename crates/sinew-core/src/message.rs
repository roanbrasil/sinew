use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Role {
    System,
    User,
    Assistant,
    Tool,
}

/// A model's request to invoke a tool.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ToolCall {
    pub id: String,
    pub name: String,
    pub arguments: serde_json::Value,
}

/// Where the bytes of an image, document or audio clip come from.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum MediaSource {
    /// A URL the provider fetches itself (`https://`, `gs://`, a Gemini
    /// Files API URI, ...). `media_type` is optional; providers that need it
    /// (Gemini) guess it from the file extension when absent.
    Url {
        url: String,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        media_type: Option<String>,
    },
    /// Inline bytes, already base64-encoded (standard alphabet, no newlines).
    Base64 { media_type: String, data: String },
}

impl MediaSource {
    /// The MIME type, given or guessed from a URL's extension.
    pub fn media_type(&self) -> Option<String> {
        match self {
            MediaSource::Base64 { media_type, .. } => Some(media_type.clone()),
            MediaSource::Url { media_type, url } => {
                media_type.clone().or_else(|| guess_media_type(url))
            }
        }
    }

    /// `data:<mime>;base64,<data>` for inline sources, the URL otherwise.
    pub fn to_url(&self) -> String {
        match self {
            MediaSource::Url { url, .. } => url.clone(),
            MediaSource::Base64 { media_type, data } => {
                format!("data:{media_type};base64,{data}")
            }
        }
    }
}

/// Image detail hint (OpenAI `detail`; ignored by other providers).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ImageDetail {
    Auto,
    Low,
    High,
}

/// A non-text (or extra text) piece of a message.
///
/// Parts follow [`Message::content`] in order: the provider sees the content
/// text first (when not empty), then each part.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ContentPart {
    Text {
        text: String,
    },
    Image {
        source: MediaSource,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        detail: Option<ImageDetail>,
    },
    /// A document such as a PDF (Anthropic `document`, Gemini
    /// `inlineData`/`fileData`, OpenAI `file`).
    Document {
        source: MediaSource,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        name: Option<String>,
    },
    /// Audio input (OpenAI `input_audio`, Gemini `inlineData`/`fileData`).
    Audio {
        source: MediaSource,
    },
}

impl ContentPart {
    pub fn text(text: impl Into<String>) -> Self {
        ContentPart::Text { text: text.into() }
    }

    /// An image the provider downloads from `url`.
    pub fn image_url(url: impl Into<String>) -> Self {
        ContentPart::Image {
            source: MediaSource::Url {
                url: url.into(),
                media_type: None,
            },
            detail: None,
        }
    }

    /// Inline image bytes; `media_type` like `image/png` or `image/jpeg`.
    pub fn image_bytes(bytes: &[u8], media_type: impl Into<String>) -> Self {
        Self::image_base64(media_type, base64_encode(bytes))
    }

    /// Inline image data that is already base64-encoded.
    pub fn image_base64(media_type: impl Into<String>, data: impl Into<String>) -> Self {
        ContentPart::Image {
            source: MediaSource::Base64 {
                media_type: media_type.into(),
                data: data.into(),
            },
            detail: None,
        }
    }

    /// Inline PDF bytes.
    pub fn pdf_bytes(bytes: &[u8]) -> Self {
        ContentPart::Document {
            source: MediaSource::Base64 {
                media_type: "application/pdf".into(),
                data: base64_encode(bytes),
            },
            name: None,
        }
    }

    /// A PDF the provider downloads from `url`.
    pub fn pdf_url(url: impl Into<String>) -> Self {
        ContentPart::Document {
            source: MediaSource::Url {
                url: url.into(),
                media_type: Some("application/pdf".into()),
            },
            name: None,
        }
    }

    /// Inline audio bytes; `media_type` like `audio/wav` or `audio/mpeg`.
    pub fn audio_bytes(bytes: &[u8], media_type: impl Into<String>) -> Self {
        ContentPart::Audio {
            source: MediaSource::Base64 {
                media_type: media_type.into(),
                data: base64_encode(bytes),
            },
        }
    }

    /// A file by URI with an explicit MIME type (e.g. a Gemini Files API URI
    /// or `gs://` object). Images become [`ContentPart::Image`], audio
    /// [`ContentPart::Audio`], anything else [`ContentPart::Document`].
    pub fn file_uri(uri: impl Into<String>, media_type: impl Into<String>) -> Self {
        let media_type = media_type.into();
        let source = MediaSource::Url {
            url: uri.into(),
            media_type: Some(media_type.clone()),
        };
        if media_type.starts_with("image/") {
            ContentPart::Image {
                source,
                detail: None,
            }
        } else if media_type.starts_with("audio/") {
            ContentPart::Audio { source }
        } else {
            ContentPart::Document { source, name: None }
        }
    }

    /// Sets the OpenAI image detail (no-op for other parts).
    pub fn with_detail(mut self, d: ImageDetail) -> Self {
        if let ContentPart::Image { detail, .. } = &mut self {
            *detail = Some(d);
        }
        self
    }

    /// Sets a document's file name / title (no-op for other parts).
    pub fn with_name(mut self, n: impl Into<String>) -> Self {
        if let ContentPart::Document { name, .. } = &mut self {
            *name = Some(n.into());
        }
        self
    }
}

/// Provider-neutral chat message.
///
/// `content` is the message text. Images, documents, audio and extra text
/// blocks go in `parts` (empty for plain text messages, and omitted from the
/// serialized form then, so text-only messages look exactly as before).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Message {
    pub role: Role,
    pub content: String,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub tool_calls: Vec<ToolCall>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_call_id: Option<String>,
    /// Multimodal parts that follow `content`.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub parts: Vec<ContentPart>,
}

impl Message {
    fn new(role: Role, content: impl Into<String>) -> Self {
        Self {
            role,
            content: content.into(),
            tool_calls: Vec::new(),
            tool_call_id: None,
            parts: Vec::new(),
        }
    }
    pub fn system(content: impl Into<String>) -> Self {
        Self::new(Role::System, content)
    }
    pub fn user(content: impl Into<String>) -> Self {
        Self::new(Role::User, content)
    }
    pub fn assistant(content: impl Into<String>) -> Self {
        Self::new(Role::Assistant, content)
    }
    pub fn assistant_with_tools(content: impl Into<String>, calls: Vec<ToolCall>) -> Self {
        Self {
            tool_calls: calls,
            ..Self::new(Role::Assistant, content)
        }
    }
    pub fn tool_result(call_id: impl Into<String>, content: impl Into<String>) -> Self {
        Self {
            tool_call_id: Some(call_id.into()),
            ..Self::new(Role::Tool, content)
        }
    }

    /// A user message with text followed by any parts.
    pub fn user_with_parts(
        content: impl Into<String>,
        parts: impl IntoIterator<Item = ContentPart>,
    ) -> Self {
        Self {
            parts: parts.into_iter().collect(),
            ..Self::new(Role::User, content)
        }
    }

    /// A user message with text and images, e.g.
    /// `Message::user_with_images("What is this?", [ContentPart::image_url(url)])`.
    pub fn user_with_images(
        content: impl Into<String>,
        images: impl IntoIterator<Item = ContentPart>,
    ) -> Self {
        Self::user_with_parts(content, images)
    }

    /// Appends a part.
    pub fn with_part(mut self, part: ContentPart) -> Self {
        self.parts.push(part);
        self
    }

    pub fn has_tool_calls(&self) -> bool {
        !self.tool_calls.is_empty()
    }

    /// Whether the message carries images, documents or audio.
    pub fn has_media(&self) -> bool {
        self.parts
            .iter()
            .any(|p| !matches!(p, ContentPart::Text { .. }))
    }

    /// `content` plus the text of any [`ContentPart::Text`] parts.
    pub fn text(&self) -> String {
        let mut out = self.content.clone();
        for p in &self.parts {
            if let ContentPart::Text { text } = p {
                if !out.is_empty() {
                    out.push('\n');
                }
                out.push_str(text);
            }
        }
        out
    }
}

/// Guesses a MIME type from a URL's file extension.
pub(crate) fn guess_media_type(url: &str) -> Option<String> {
    let path = url.split(['?', '#']).next().unwrap_or(url);
    let ext = path.rsplit_once('.')?.1.to_ascii_lowercase();
    let mt = match ext.as_str() {
        "png" => "image/png",
        "jpg" | "jpeg" => "image/jpeg",
        "gif" => "image/gif",
        "webp" => "image/webp",
        "heic" => "image/heic",
        "pdf" => "application/pdf",
        "txt" => "text/plain",
        "md" => "text/markdown",
        "html" | "htm" => "text/html",
        "csv" => "text/csv",
        "wav" => "audio/wav",
        "mp3" => "audio/mpeg",
        "flac" => "audio/flac",
        "ogg" => "audio/ogg",
        "m4a" => "audio/mp4",
        "aac" => "audio/aac",
        "mp4" => "video/mp4",
        "webm" => "video/webm",
        "mov" => "video/quicktime",
        _ => return None,
    };
    Some(mt.to_string())
}

/// Standard base64 with padding (RFC 4648).
pub fn base64_encode(bytes: &[u8]) -> String {
    const T: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    let mut out = String::with_capacity(bytes.len().div_ceil(3) * 4);
    let mut chunks = bytes.chunks_exact(3);
    for c in &mut chunks {
        let n = (u32::from(c[0]) << 16) | (u32::from(c[1]) << 8) | u32::from(c[2]);
        for shift in [18, 12, 6, 0] {
            out.push(T[(n >> shift) as usize & 63] as char);
        }
    }
    let rem = chunks.remainder();
    if !rem.is_empty() {
        let b1 = rem.get(1).copied().unwrap_or(0);
        let n = (u32::from(rem[0]) << 16) | (u32::from(b1) << 8);
        out.push(T[(n >> 18) as usize & 63] as char);
        out.push(T[(n >> 12) as usize & 63] as char);
        out.push(if rem.len() == 2 {
            T[(n >> 6) as usize & 63] as char
        } else {
            '='
        });
        out.push('=');
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn base64_matches_rfc4648_vectors() {
        for (i, o) in [
            ("", ""),
            ("f", "Zg=="),
            ("fo", "Zm8="),
            ("foo", "Zm9v"),
            ("foob", "Zm9vYg=="),
            ("fooba", "Zm9vYmE="),
            ("foobar", "Zm9vYmFy"),
        ] {
            assert_eq!(base64_encode(i.as_bytes()), o);
        }
    }

    #[test]
    fn text_messages_serialize_without_parts() {
        let m = Message::user("hi");
        let v = serde_json::to_value(&m).unwrap();
        assert!(v.get("parts").is_none());
        // Old payloads (no `parts`) still deserialize.
        let back: Message = serde_json::from_str(r#"{"role":"user","content":"hi"}"#).unwrap();
        assert_eq!(back, m);
    }

    #[test]
    fn multimodal_round_trip_and_helpers() {
        let m = Message::user_with_images(
            "what is it?",
            [ContentPart::image_url("https://x/cat.PNG?s=1").with_detail(ImageDetail::Low)],
        )
        .with_part(ContentPart::pdf_bytes(b"%PDF").with_name("a.pdf"))
        .with_part(ContentPart::text("extra"));
        assert!(m.has_media());
        assert_eq!(m.text(), "what is it?\nextra");
        let json = serde_json::to_string(&m).unwrap();
        let back: Message = serde_json::from_str(&json).unwrap();
        assert_eq!(back, m);
        match &m.parts[0] {
            ContentPart::Image { source, .. } => {
                assert_eq!(source.media_type().as_deref(), Some("image/png"))
            }
            _ => panic!(),
        }
        assert_eq!(
            ContentPart::image_bytes(b"abc", "image/png"),
            ContentPart::image_base64("image/png", "YWJj")
        );
        assert!(matches!(
            ContentPart::file_uri("gs://b/a.mp3", "audio/mpeg"),
            ContentPart::Audio { .. }
        ));
    }
}
