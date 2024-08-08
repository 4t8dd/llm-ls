use super::{Generation, NAME, VERSION};
use custom_types::llm_ls::{Backend, Ide};
use reqwest::header::{HeaderMap, HeaderValue, AUTHORIZATION, USER_AGENT};
use serde::{Deserialize, Serialize};
use serde_json::{json, Map, Value};
use std::fmt::Display;

use crate::error::{Error, Result};

#[derive(Debug, Deserialize)]
pub struct APIError {
    error: String,
}

impl std::error::Error for APIError {
    fn description(&self) -> &str {
        &self.error
    }
}

impl Display for APIError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.error)
    }
}

#[derive(Debug, Deserialize)]
#[serde(untagged)]
pub(crate) enum APIResponse {
    Generation(Generation),
    Generations(Vec<Generation>),
    Error(APIError),
}

fn build_tgi_headers(api_token: Option<&String>, ide: Ide) -> Result<HeaderMap> {
    let mut headers = HeaderMap::new();
    let user_agent = format!("{NAME}/{VERSION}; rust/unknown; ide/{ide:?}");
    headers.insert(USER_AGENT, HeaderValue::from_str(&user_agent)?);

    if let Some(api_token) = api_token {
        headers.insert(
            AUTHORIZATION,
            HeaderValue::from_str(&format!("Bearer {api_token}"))?,
        );
    }

    Ok(headers)
}

fn parse_tgi_text(text: &str) -> Result<Vec<Generation>> {
    match serde_json::from_str(text)? {
        APIResponse::Generation(gen) => Ok(vec![gen]),
        APIResponse::Generations(_) => Err(Error::InvalidBackend),
        APIResponse::Error(err) => Err(Error::Tgi(err)),
    }
}

fn build_api_headers(api_token: Option<&String>, ide: Ide) -> Result<HeaderMap> {
    build_tgi_headers(api_token, ide)
}

fn parse_api_text(text: &str) -> Result<Vec<Generation>> {
    match serde_json::from_str(text)? {
        APIResponse::Generation(gen) => Ok(vec![gen]),
        APIResponse::Generations(gens) => Ok(gens),
        APIResponse::Error(err) => Err(Error::InferenceApi(err)),
    }
}

#[derive(Debug, Serialize, Deserialize)]
struct LlamaCppGeneration {
    content: String,
}

impl From<LlamaCppGeneration> for Generation {
    fn from(value: LlamaCppGeneration) -> Self {
        Generation {
            generated_text: value.content,
        }
    }
}

#[derive(Debug, Deserialize)]
#[serde(untagged)]
enum LlamaCppAPIResponse {
    Generation(LlamaCppGeneration),
    Error(APIError),
}

fn build_llamacpp_headers() -> HeaderMap {
    HeaderMap::new()
}

fn parse_llamacpp_text(text: &str) -> Result<Vec<Generation>> {
    match serde_json::from_str(text)? {
        LlamaCppAPIResponse::Generation(gen) => Ok(vec![gen.into()]),
        LlamaCppAPIResponse::Error(err) => Err(Error::LlamaCpp(err)),
    }
}

#[derive(Debug, Serialize, Deserialize)]
struct OllamaGeneration {
    response: String,
}

impl From<OllamaGeneration> for Generation {
    fn from(value: OllamaGeneration) -> Self {
        Generation {
            generated_text: value.response,
        }
    }
}

#[derive(Debug, Deserialize)]
#[serde(untagged)]
enum OllamaAPIResponse {
    Generation(OllamaGeneration),
    Error(APIError),
}

fn build_ollama_headers() -> HeaderMap {
    HeaderMap::new()
}

fn parse_ollama_text(text: &str) -> Result<Vec<Generation>> {
    match serde_json::from_str(text)? {
        OllamaAPIResponse::Generation(gen) => Ok(vec![gen.into()]),
        OllamaAPIResponse::Error(err) => Err(Error::Ollama(err)),
    }
}

#[derive(Debug, Deserialize, Serialize)]
struct OpenAIResponse {
    choices: Vec<Choice>,
}

#[derive(Debug, Deserialize, Serialize)]
struct Choice {
    index: u32,
    finish_reason: String,
    message: OpenAIMessage,
}

#[derive(Debug, Deserialize, Serialize)]
struct OpenAIMessage {
    role: String,
    content: String,
}

#[derive(Debug, Deserialize, Serialize)]
struct Delta {
    content: String,
}

#[derive(Debug, Deserialize, Serialize)]
struct OpenAIStreamMessage {
    delta: Delta,
    index: u32,
    finish_reason: String,
}

#[derive(Debug, Deserialize, Serialize)]
struct OpenAIStreamResponse {
    id: String,
    object: String,
    model: String,
    created: u32,
    choices: Vec<OpenAIStreamMessage>,
}

impl From<Choice> for Generation {
    fn from(value: Choice) -> Self {
        Generation {
            generated_text: value.message.content,
        }
    }
}

#[derive(Debug, Deserialize)]
#[serde(untagged)]
enum OpenAIErrorLoc {
    String(String),
    Int(u32),
}

impl Display for OpenAIErrorLoc {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            OpenAIErrorLoc::String(s) => s.fmt(f),
            OpenAIErrorLoc::Int(i) => i.fmt(f),
        }
    }
}

#[derive(Debug, Deserialize)]
struct OpenAIErrorDetail {
    loc: OpenAIErrorLoc,
    msg: String,
    r#type: String,
}

#[derive(Debug, Deserialize)]
pub struct OpenAIError {
    detail: Vec<OpenAIErrorDetail>,
}

impl Display for OpenAIError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        for (i, item) in self.detail.iter().enumerate() {
            if i != 0 {
                writeln!(f)?;
            }
            write!(f, "{}: {} ({})", item.loc, item.msg, item.r#type)?;
        }
        Ok(())
    }
}

#[derive(Debug, Deserialize)]
#[serde(untagged)]
enum OpenAIAPIResponse {
    Generation(OpenAIResponse),
    Error(OpenAIError),
}

#[derive(Debug, Deserialize)]
pub struct ClaudeError {
    r#type: String,
    message: String,
}

impl Display for ClaudeError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Claude API Error: {} (error typ {}", self.message, self.r#type)
    }
}

#[derive(Debug, Deserialize)]
struct ClaudeMessage {
    message: String,
    r#type: String,
}
#[derive(Debug, Deserialize)]
pub struct ClaudeResponse {
    content: Vec<ClaudeMessage>,
}

#[derive(Debug, Deserialize)]
#[serde(untagged)]
pub enum  ClaudeAPIResponse {
    Generation(ClaudeResponse),
    Error(ClaudeError),
}

impl From<ClaudeMessage> for Generation {
    fn from(value: ClaudeMessage) -> Generation {
        Generation{
            generated_text: value.message
        }
    }
}

fn build_claude_headers(api_token: Option<&String>, ide: Ide) -> Result<HeaderMap> {
    let mut headers = HeaderMap::new();
    let user_agent = format!("{NAME}/{VERSION}; rust/unknown; ide/{ide:?}");
    headers.insert(USER_AGENT, HeaderValue::from_str(&user_agent)?);

    if let Some(api_token) = api_token {
        headers.insert(
            "x-api-key",
            HeaderValue::from_str(&format!("{api_token}"))?,
        );
    }

    Ok(headers)
}

fn parse_claude_text(text: &str) -> Result<Vec<Generation>> {
    match serde_json::from_str(text)? {
        ClaudeAPIResponse::Generation(result) =>{
            Ok(result.content.into_iter().map(|content| content.into()).collect())
        }
        ClaudeAPIResponse::Error(err) =>{
            Err(Error::Claude(err))
        }
    }
}


#[derive(Debug, Deserialize)]
pub struct GeminiError {
    code: u32,
    message: String,
    status: String,
}
impl Display for GeminiError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Gemini API Error: {} (Code: {}, Status: {})", self.message, self.code, self.status)
    }
}


#[derive(Debug, Deserialize)]
struct Candidate {
    content: Content,
}

#[derive(Debug, Deserialize)]
struct GeminiResponse {
    candidates: Vec<Candidate>,
}

#[derive(Debug, Deserialize)]
struct Content {
    parts: Vec<Part>,
    role: String,
}

#[derive(Debug, Deserialize)]
struct Part {
    text: String,
}

#[derive(Debug, Deserialize)]
#[serde(untagged)]
enum GeminiAPIResponse {
    Generation(GeminiResponse),
    Error(GeminiError),
}

impl From<Candidate> for Generation {
    fn from(value: Candidate) -> Self {
        let text: String = value.content.parts
            .into_iter()
            .map(|part| part.text)
            .collect::<Vec<String>>()
            .join("");
        Generation {
            generated_text: text,
        }
    }
}

fn build_gemini_headers(api_token: Option<&String>, ide: Ide) -> Result<HeaderMap> {
    let mut headers = HeaderMap::new();
    let user_agent = format!("{NAME}/{VERSION}; rust/unknown; ide/{ide:?}");
    headers.insert(USER_AGENT, HeaderValue::from_str(&user_agent)?);

    if let Some(api_token) = api_token {
        headers.insert(
            "x-goog-api-key",
            HeaderValue::from_str(&format!("{api_token}"))?,
        );
    }

    Ok(headers)
}

fn parse_gemini_text(text: &str) -> Result<Vec<Generation>> {
    match serde_json::from_str(text)? {
        GeminiAPIResponse::Generation(result) => {
            Ok(result.candidates.into_iter().map(|x| x.into()).collect())
        }
        GeminiAPIResponse::Error(err) => {
            Err(Error::Gemini(err))
        }
    }
}

fn build_openai_headers(api_token: Option<&String>, ide: Ide) -> Result<HeaderMap> {
    build_api_headers(api_token, ide)
}

fn parse_openai_text(text: &str) -> Result<Vec<Generation>> {
    match serde_json::from_str(text)? {
        OpenAIAPIResponse::Generation(completion) => {
            Ok(completion.choices.into_iter().map(|x| x.into()).collect())
        }
        OpenAIAPIResponse::Error(err) => Err(Error::OpenAI(err)),
    }
}

pub(crate) fn build_body(
    backend: &Backend,
    model: String,
    prompt: String,
    mut request_body: Map<String, Value>,
) -> Map<String, Value> {
    match backend {
        Backend::HuggingFace { .. } | Backend::Tgi { .. } => {
            request_body.insert("inputs".to_owned(), Value::String(prompt));
            if let Some(Value::Object(params)) = request_body.get_mut("parameters") {
                params.insert("return_full_text".to_owned(), Value::Bool(false));
            } else {
                let params = json!({ "parameters": { "return_full_text": false } });
                request_body.insert("parameters".to_owned(), params);
            }
        }
        Backend::LlamaCpp { .. }| Backend::Claude { url } => {
            request_body.insert("prompt".to_owned(), Value::String(prompt));
        }
        Backend::Gemini { .. } => {
            let content = json!({
                "role": "user",
                "parts": [{
                    "text": prompt
                }]
            });
            request_body.insert("contents".to_owned(), Value::Array(vec![Value::Object(content.as_object().unwrap().clone())]));
        }
        Backend::Ollama { .. } | Backend::OpenAi { .. } => {
            let mut message = Map::new();
            message.insert("role".to_owned(), Value::String("user".to_owned()));
            message.insert("content".to_owned(), Value::String(prompt.to_owned()));

            request_body
                .entry("messages")
                .and_modify(|msgs| {
                    let array = msgs
                        .as_array_mut()
                        .expect("Expected an array for 'messages' field");
                    array.push(Value::Object(message.clone()));
                })
                .or_insert_with(|| Value::Array(vec![Value::Object(message)]));
            request_body.insert("model".to_owned(), Value::String(model));
            request_body.insert("stream".to_owned(), Value::Bool(false));
        }
    };
    request_body
}

pub(crate) fn build_headers(
    backend: &Backend,
    api_token: Option<&String>,
    ide: Ide,
) -> Result<HeaderMap> {
    match backend {
        Backend::HuggingFace { .. } => build_api_headers(api_token, ide),
        Backend::LlamaCpp { .. } => Ok(build_llamacpp_headers()),
        Backend::Ollama { .. } => Ok(build_ollama_headers()),
        Backend::OpenAi { .. } => build_openai_headers(api_token, ide),
        Backend::Tgi { .. } => build_tgi_headers(api_token, ide),
        Backend::Gemini { .. } => build_gemini_headers(api_token, ide),
        Backend::Claude {..} => build_claude_headers(api_token, ide),
    }
}

pub(crate) fn parse_generations(backend: &Backend, text: &str) -> Result<Vec<Generation>> {
    match backend {
        Backend::HuggingFace { .. } => parse_api_text(text),
        Backend::LlamaCpp { .. } => parse_llamacpp_text(text),
        Backend::Ollama { .. } => parse_ollama_text(text),
        Backend::OpenAi { .. } => parse_openai_text(text),
        Backend::Tgi { .. } => parse_tgi_text(text),
        Backend::Gemini { .. } => parse_gemini_text(text),
        Backend::Claude { .. } => parse_claude_text(text),
    }
}
