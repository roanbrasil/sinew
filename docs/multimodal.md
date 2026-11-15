# Multimodal messages

A `Message` keeps its text in `content: String`, as before. Images,
documents, audio and extra text blocks go in `parts: Vec<ContentPart>`.
`parts` is empty for text-only messages and is left out of the serialized
form, so existing code, stored conversations and checkpoints are unchanged.

```rust
use sinew_core::{ContentPart, ImageDetail, Message};

let msg = Message::user_with_images(
    "What is in this picture, and does it match the PDF?",
    [
        ContentPart::image_url("https://example.com/cat.png").with_detail(ImageDetail::High),
        ContentPart::image_bytes(&std::fs::read("cat.jpg")?, "image/jpeg"),
    ],
)
.with_part(ContentPart::pdf_bytes(&std::fs::read("report.pdf")?).with_name("report.pdf"));
```

The provider sees `content` first (when it is not empty), then each part in
order.

## Parts

| Constructor | Part |
|---|---|
| `ContentPart::text(s)` | Extra text block. |
| `ContentPart::image_url(url)` | Image the provider downloads. |
| `ContentPart::image_bytes(&bytes, mime)` | Inline image, base64-encoded for you. |
| `ContentPart::image_base64(mime, data)` | Inline image that is already base64. |
| `ContentPart::pdf_bytes(&bytes)` / `pdf_url(url)` | PDF document. |
| `ContentPart::audio_bytes(&bytes, mime)` | Inline audio (`audio/wav`, `audio/mpeg`, ...). |
| `ContentPart::file_uri(uri, mime)` | A file by URI with an explicit MIME type, such as a Gemini Files API URI, a `gs://` object or an uploaded file id. Image, audio or document is picked from the MIME type. |

Modifiers: `.with_detail(ImageDetail::Low | High | Auto)` sets the OpenAI
image detail, and `.with_name(..)` sets a document's file name or title.

`MediaSource` is either `Url { url, media_type }` or
`Base64 { media_type, data }`. `Message::has_media()` and `Message::text()`
(content plus text parts) help code that inspects messages.

Message constructors: `Message::user_with_parts(text, parts)`,
`Message::user_with_images(text, images)` and `msg.with_part(part)`. Parts on
tool results are supported too, for example a screenshot returned by a tool:
`Message::tool_result(id, "done").with_part(ContentPart::image_bytes(..))`.

## Provider encoding

| Part | OpenAI-compatible | Anthropic | Gemini |
|---|---|---|---|
| text | `{"type":"text"}` | `{"type":"text"}` | `{"text"}` |
| image URL | `image_url.url` | `image` with `source.type = "url"` | `fileData {mimeType, fileUri}` (MIME guessed from the extension) |
| image bytes | `image_url.url = "data:<mime>;base64,..."` | `image` with `source.type = "base64"` | `inlineData {mimeType, data}` |
| PDF bytes | `{"type":"file","file":{"filename","file_data"}}` | `document` with base64 source (+ `title`) | `inlineData` |
| PDF URL | uploaded id (`file-...`) as `file_id`, otherwise a text reference | `document` with URL source; Files API ids (`file_...`) as `{"type":"file"}` | `fileData` |
| audio bytes | `{"type":"input_audio","input_audio":{"data","format"}}` | not supported: sent as a text note | `inlineData` |
| audio URL | text reference | text note | `fileData` |

Text-only messages encode exactly as before. OpenAI-compatible servers get
`content` as a plain string, Anthropic and Gemini a single text block. The
Anthropic and Gemini system prompts accept only text, so those providers
send a system message's text (content plus text parts) and drop its media.

Unsupported combinations are never dropped silently. They become a short text
reference such as `[audio: https://...]`, so the request still succeeds and
the model knows something was attached. Local OpenAI-compatible servers
(Ollama, vLLM, LM Studio, llama.cpp) accept the same image format as OpenAI
when the loaded model has vision support.

## Tips

- Base64 makes a payload about 33% larger than the raw file. For large or
  repeated files, use URLs, the Anthropic Files API or the Gemini Files API,
  and pass the id or URI with `ContentPart::file_uri`.
- For rate limiting, `estimate_request_tokens` counts a flat allowance per
  part: about 800 tokens per image, 1,500 per document and 1,000 per audio
  clip. Real usage comes back in `Usage`.
