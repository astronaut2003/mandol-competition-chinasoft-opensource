from dataclasses import dataclass, field, asdict
from pathlib import Path
import os
from urllib.parse import urlsplit
from mandol_substrate.state.models import digest

ROOT = Path(os.getenv('CHINASOFT_APP_ROOT', str(Path(__file__).resolve().parents[3]))).resolve()


def env(name, default):
    return os.getenv('CHINASOFT_' + name, str(default))


@dataclass
class Settings:
    data_dir: Path = field(default_factory=lambda: Path(env('DATA_DIR', ROOT / 'data')).resolve())
    host: str = field(default_factory=lambda: env('HOST', '127.0.0.1'))
    port: int = field(default_factory=lambda: int(env('PORT', 20600)))
    qdrant_url: str = field(default_factory=lambda: env('QDRANT_URL', 'http://127.0.0.1:20633'))
    collection: str = field(default_factory=lambda: env('QDRANT_COLLECTION', 'chinasoft_opensource_memory_v1'))
    llm_base_url: str = field(default_factory=lambda: env('LLM_BASE_URL', 'https://api.deepseek.com/v1'))
    llm_api_key_env: str = field(default_factory=lambda: env('LLM_API_KEY_ENV', 'DEEPSEEK_API_KEY'))
    llm_model: str = field(default_factory=lambda: env('LLM_MODEL', 'deepseek-flash'))
    llm_request_model: str = field(default_factory=lambda: env('LLM_REQUEST_MODEL', 'deepseek-flash'))
    vlm_model: str = field(default_factory=lambda: env('VLM_MODEL', 'deepseek-flash'))
    vlm_request_model: str = field(default_factory=lambda: env('VLM_REQUEST_MODEL', 'deepseek-flash'))
    llm_timeout: float = field(default_factory=lambda: float(env('LLM_TIMEOUT', 90)))
    llm_max_concurrency: int = field(default_factory=lambda: int(env('LLM_MAX_CONCURRENCY', 2)))
    llm_max_tokens: int = field(default_factory=lambda: int(env('LLM_MAX_TOKENS', 4096)))
    json_output_mode: str = field(default_factory=lambda: env('JSON_OUTPUT_MODE', 'json_object'))
    generation_retry_policy: int = field(default_factory=lambda: int(env('GENERATION_RETRY_POLICY', 3)))
    accepted_response_models: str = field(default_factory=lambda: env('LLM_ACCEPTED_RESPONSE_MODELS', 'deepseek-flash'))
    reasoning_protocol: str = field(default_factory=lambda: env('REASONING_PROTOCOL','auto'))
    search_enabled: bool = field(default_factory=lambda: env('SEARCH_ENABLED','false').lower()=='true')
    search_base_url: str = field(default_factory=lambda: env('SEARCH_BASE_URL','https://api.deepseek.com/v1'))
    search_api_key_env: str = field(default_factory=lambda: env('SEARCH_API_KEY_ENV','DEEPSEEK_API_KEY'))
    search_timeout: float = field(default_factory=lambda: float(env('SEARCH_TIMEOUT',12)))
    search_max_queries: int = field(default_factory=lambda: int(env('SEARCH_MAX_QUERIES',2)))
    search_planner_enabled: bool = field(default_factory=lambda: env('SEARCH_PLANNER_ENABLED','true').lower()=='true')
    search_planner_timeout: float = field(default_factory=lambda: float(env('SEARCH_PLANNER_TIMEOUT',4)))
    search_max_results: int = field(default_factory=lambda: int(env('SEARCH_MAX_RESULTS',5)))
    search_max_fetch: int = field(default_factory=lambda: int(env('SEARCH_MAX_FETCH',0)))
    search_context_chars: int = field(default_factory=lambda: int(env('SEARCH_CONTEXT_CHARS',10000)))
    search_fetch_hosts: str = field(default_factory=lambda: env('SEARCH_FETCH_HOSTS','api-docs.deepseek.com,docs.python.org'))
    web_requests_per_day: int = field(default_factory=lambda: int(env('WEB_REQUESTS_PER_DAY',1000)))
    web_workspace_per_day: int = field(default_factory=lambda: int(env('WEB_WORKSPACE_PER_DAY',100)))
    vlm_accepted_response_models: str = field(default_factory=lambda: env('VLM_ACCEPTED_RESPONSE_MODELS', 'deepseek-flash'))
    retrieval_backend: str = field(default_factory=lambda: env('RETRIEVAL_BACKEND', 'vllm'))
    embedding_url: str = field(default_factory=lambda: env('EMBEDDING_BASE_URL', 'http://127.0.0.1:20631/v1'))
    embedding_api_key_env: str = field(default_factory=lambda: env('EMBEDDING_API_KEY_ENV', ''))
    embedding_model: str = field(default_factory=lambda: env('EMBEDDING_MODEL', 'Qwen/Qwen3-Embedding-0.6B'))
    embedding_dimension: int = field(default_factory=lambda: int(env('EMBEDDING_DIMENSION', 1024)))
    reranker_url: str = field(default_factory=lambda: env('RERANKER_BASE_URL', 'http://127.0.0.1:20632'))
    reranker_api_key_env: str = field(default_factory=lambda: env('RERANKER_API_KEY_ENV', ''))
    reranker_model: str = field(default_factory=lambda: env('RERANKER_MODEL', 'BAAI/bge-reranker-v2-m3'))
    sparse_url: str = field(default_factory=lambda: env('SPLADE_URL', 'http://127.0.0.1:20636'))
    sparse_api_key_env: str = field(default_factory=lambda: env('SPLADE_API_KEY_ENV', ''))
    dense_timeout: float = field(default_factory=lambda: float(env('DENSE_TIMEOUT', 30)))
    sparse_timeout: float = field(default_factory=lambda: float(env('SPARSE_TIMEOUT', 30)))
    reranker_timeout: float = field(default_factory=lambda: float(env('RERANKER_TIMEOUT', 30)))
    dense_concurrency: int = field(default_factory=lambda: int(env('DENSE_CONCURRENCY', 2)))
    sparse_concurrency: int = field(default_factory=lambda: int(env('SPARSE_CONCURRENCY', 2)))
    reranker_concurrency: int = field(default_factory=lambda: int(env('RERANKER_CONCURRENCY', 2)))
    allow_retrieval_degraded: bool = field(default_factory=lambda: env('ALLOW_RETRIEVAL_DEGRADED', 'false').lower() == 'true')
    memory_query_strategy: str = field(default_factory=lambda: env('MEMORY_QUERY_STRATEGY', 'adaptive_v33'))
    chat_history_chars: int = field(default_factory=lambda: int(env('CHAT_HISTORY_CHARS', '16000')))
    chat_attachment_chars: int = field(default_factory=lambda: int(env('CHAT_ATTACHMENT_CHARS', '16000')))
    attachment_max_bytes: int = field(default_factory=lambda: int(env('ATTACHMENT_MAX_BYTES', '262144')))
    attachment_max_files: int = field(default_factory=lambda: int(env('ATTACHMENT_MAX_FILES', '4')))
    queue_capacity: int = field(default_factory=lambda: int(env('QUEUE_CAPACITY', 192)))
    worker_max_attempts: int = field(default_factory=lambda: int(env('WORKER_MAX_ATTEMPTS', 3)))
    max_media_bytes: int = 8 * 1024 * 1024
    secure_cookie: bool = field(default_factory=lambda: env('SECURE_COOKIE', 'false').lower() == 'true')
    registration_enabled: bool = field(default_factory=lambda: env('REGISTRATION_ENABLED', 'false').lower() == 'true')
    registration_approval: bool = field(default_factory=lambda: env('REGISTRATION_APPROVAL', 'true').lower() == 'true')
    session_ttl: int = field(default_factory=lambda: int(env('SESSION_TTL', 604800)))
    account_job_quota: int = field(default_factory=lambda: int(env('ACCOUNT_JOB_QUOTA', 96)))
    max_accounts: int = field(default_factory=lambda: int(env('MAX_ACCOUNTS', 100)))
    quota_workspaces: int = field(default_factory=lambda: int(env('QUOTA_WORKSPACES', 10)))
    quota_conversations: int = field(default_factory=lambda: int(env('QUOTA_CONVERSATIONS', 300)))
    quota_repositories: int = field(default_factory=lambda: int(env('QUOTA_REPOSITORIES', 5)))
    quota_storage_bytes: int = field(default_factory=lambda: int(env('QUOTA_STORAGE_BYTES', 100*1024*1024)))
    quota_chat_per_minute: int = field(default_factory=lambda: int(env('QUOTA_CHAT_PER_MINUTE', 10)))
    quota_model_per_day: int = field(default_factory=lambda: int(env('QUOTA_MODEL_PER_DAY', 300)))
    model_requests_per_day: int = field(default_factory=lambda: int(env('MODEL_REQUESTS_PER_DAY', 2000)))
    remote_git_enabled: bool = field(default_factory=lambda: env('REMOTE_GIT_ENABLED', 'true').lower() == 'true')
    remote_git_max_files: int = field(default_factory=lambda: int(env('REMOTE_GIT_MAX_FILES', 500)))
    remote_git_max_bytes: int = field(default_factory=lambda: int(env('REMOTE_GIT_MAX_BYTES', 20*1024*1024)))
    remote_git_timeout: int = field(default_factory=lambda: int(env('REMOTE_GIT_TIMEOUT', 180)))
    remote_git_cache_bytes: int = field(default_factory=lambda: int(env('REMOTE_GIT_CACHE_BYTES', 256*1024*1024)))
    demo_enabled: bool = field(default_factory=lambda: env('DEMO_ENABLED', 'false').lower() == 'true')
    repository_roots: str = field(default_factory=lambda: env('REPOSITORY_ROOTS', ''))
    repository_excludes: str = field(default_factory=lambda: env('REPOSITORY_EXCLUDES', '.release-private/*'))
    repository_max_files: int = field(default_factory=lambda: int(env('REPOSITORY_MAX_FILES', 2500)))
    repository_max_file_bytes: int = field(default_factory=lambda: int(env('REPOSITORY_MAX_FILE_BYTES', 256*1024)))
    repository_max_total_bytes: int = field(default_factory=lambda: int(env('REPOSITORY_MAX_TOTAL_BYTES', 12*1024*1024)))
    repository_max_chunks: int = field(default_factory=lambda: int(env('REPOSITORY_MAX_CHUNKS', 6000)))

    def __post_init__(self):
        if not 1<=self.search_max_queries<=3 or not 1<=self.search_planner_timeout<=8:
            raise ValueError('Invalid bounded automatic web query configuration')
        if self.reasoning_protocol not in ('auto','deepseek','none'):raise ValueError('Invalid reasoning protocol')
        if not 1<=self.search_timeout<=30 or not 1<=self.search_max_results<=8 or not 0<=self.search_max_fetch<=1 or not 1000<=self.search_context_chars<=12000:
            raise ValueError('Invalid bounded web tool configuration')
        if not 1<=self.web_requests_per_day<=10000 or not 1<=self.web_workspace_per_day<=1000:raise ValueError('Invalid web tool quota')
        u=urlsplit(self.search_base_url)
        if u.scheme!='https' or not u.hostname or u.username or u.password or u.query or u.fragment:raise ValueError('Search provider requires a credential-free HTTPS origin')
        if self.memory_query_strategy not in ('baseline','typed','typed_graph','typed_cost','adaptive_v33'):
            raise ValueError('Unknown memory query strategy; baseline is the rollback path')
        if not 1 <= self.max_accounts <= 10000 or not 1 <= self.remote_git_max_files <= 1000 or not 1024 <= self.remote_git_max_bytes <= 32*1024*1024:
            raise ValueError('Invalid public registration or remote Git budgets')
        if not 15 <= self.remote_git_timeout <= 600 or not 32*1024*1024 <= self.remote_git_cache_bytes <= 1024*1024*1024:
            raise ValueError('Invalid remote Git timeout/cache budget')
        if not 1 <= self.model_requests_per_day <= 100000:
            raise ValueError('Invalid platform model request budget')
        if self.retrieval_backend not in ('vllm', 'legacy-http'):
            raise ValueError('Explicit retrieval backend required: vllm or legacy-http')
        if any(not 1 <= n <= 120 for n in (self.dense_timeout, self.sparse_timeout, self.reranker_timeout)) or any(not 1 <= n <= 4 for n in (self.dense_concurrency, self.sparse_concurrency, self.reranker_concurrency)):
            raise ValueError('Retrieval timeout/concurrency must be bounded')
        if self.retrieval_backend == 'vllm' and (self.embedding_model != 'Qwen/Qwen3-Embedding-0.6B' or self.embedding_dimension != 1024 or self.reranker_model != 'BAAI/bge-reranker-v2-m3'):
            raise ValueError('The qualified vLLM profile requires Qwen3-Embedding-0.6B/1024 and BGE-reranker-v2-m3')
        if not 4000 <= self.chat_history_chars <= 24000 or not 2000 <= self.chat_attachment_chars <= 24000:
            raise ValueError('Invalid chat context budgets')
        if not 1024 <= self.attachment_max_bytes <= 262144 or not 1 <= self.attachment_max_files <= 4:
            raise ValueError('Invalid attachment budgets')
        if not self.collection.startswith('chinasoft_'):
            raise ValueError('An independent chinasoft_ collection is required')
        if not 300 <= self.session_ttl <= 2592000 or not 3 <= self.account_job_quota <= 192:
            raise ValueError('Invalid account resource settings')
        if self.port in (18600, 18601, 18602, 18610, 18633):
            raise ValueError('Reserved AML port')
        if not 1 <= self.llm_max_concurrency <= 4 or not 3 <= self.queue_capacity <= 3000:
            raise ValueError('Invalid bounded resource settings')
        if not 1 <= self.generation_retry_policy <= 4 or not 1 <= self.worker_max_attempts <= 4:
            raise ValueError('Retries must be bounded')
        if not 1 <= self.llm_timeout <= 180 or not 256 <= self.llm_max_tokens <= 12000:
            raise ValueError('Invalid generation budget')
        if self.json_output_mode not in ('json_object', 'json_schema', 'prompt'):
            raise ValueError('Invalid JSON output mode')
        if not 1 <= self.repository_max_files <= 10000 or not 1024 <= self.repository_max_file_bytes <= 1024*1024:
            raise ValueError('Repository limits must be bounded')
        if not 1024 <= self.repository_max_total_bytes <= 64*1024*1024 or not 1 <= self.repository_max_chunks <= 20000:
            raise ValueError('Repository size limits must be bounded')
        for url in (self.llm_base_url, self.embedding_url, self.reranker_url, self.qdrant_url, *([self.sparse_url] if self.sparse_url else [])):
            u = urlsplit(url)
            if u.scheme not in ('http', 'https') or not u.hostname or u.username or u.password or u.query:
                raise ValueError('Credential-free HTTP endpoints required')

    def contract(self, vision=False):
        return dict(schema='generation-binding-v1', provider=urlsplit(self.llm_base_url).hostname,
                    logical_model=self.vlm_model if vision else self.llm_model,
                    request_model=self.vlm_request_model if vision else self.llm_request_model,
                    accepted_response_models=(self.vlm_accepted_response_models if vision else self.accepted_response_models).split(','))

    def reasoning_capabilities(self,vision=False):
        model=self.vlm_request_model if vision else self.llm_request_model
        supported=self.reasoning_protocol=='deepseek' or self.reasoning_protocol=='auto' and urlsplit(self.llm_base_url).hostname=='api.deepseek.com' and model in ('deepseek-flash','deepseek-v4-pro')
        return {'modes':['fast','standard','deep'] if supported else ['fast'],'protocol':'deepseek' if supported else 'none',
                'reason':'official_protocol' if supported and urlsplit(self.llm_base_url).hostname=='api.deepseek.com' else 'explicit_compatible_provider' if supported else 'provider_not_qualified',
                'model':model,'mapping':{'fast':{'thinking':{'type':'disabled'}},'standard':{'thinking':{'type':'enabled'},'reasoning_effort':'high'},'deep':{'thinking':{'type':'enabled'},'reasoning_effort':'max'}} if supported else {'fast':{}}}

    @property
    def generation_fingerprint(self):
        return digest([self.contract(), self.contract(True), self.llm_base_url,
                       self.json_output_mode, self.llm_max_tokens, 'chinasoft-generation-v1'])

    def public(self):
        return {'llm': self.contract(), 'vlm': self.contract(True), 'embedding_model': self.embedding_model,
                'reranker_model': self.reranker_model, 'sparse_model': 'naver/splade-v3' if self.sparse_url else '显式词项降级（非 SPLADE）',
                'llm_concurrency': self.llm_max_concurrency, 'queue_capacity': self.queue_capacity,
                'json_output_mode': self.json_output_mode, 'generation_fingerprint': self.generation_fingerprint,
                'repository_roots': [str(Path(p).expanduser().resolve()) for p in self.repository_roots.split(os.pathsep) if p],
                'repository_limits': {'files':self.repository_max_files,'file_bytes':self.repository_max_file_bytes,'total_bytes':self.repository_max_total_bytes,'chunks':self.repository_max_chunks},
                'llm_configured': bool(os.getenv(self.llm_api_key_env)),
                'reasoning':self.reasoning_capabilities(),
                'web_search':{'enabled':self.search_enabled,'configured':bool(os.getenv(self.search_api_key_env)),'max_results':self.search_max_results,'max_fetch':self.search_max_fetch,
                              'max_queries':self.search_max_queries,'automatic_planning':self.search_planner_enabled},
                'retrieval_backend': self.retrieval_backend,
                'memory_query_strategy': self.memory_query_strategy,
                'retrieval_limits': {'dense': {'timeout': self.dense_timeout, 'concurrency': self.dense_concurrency},
                                     'sparse': {'timeout': self.sparse_timeout, 'concurrency': self.sparse_concurrency},
                                     'reranker': {'timeout': self.reranker_timeout, 'concurrency': self.reranker_concurrency}},
                'embedding_configured': bool(self.embedding_url) if self.retrieval_backend == 'vllm' else bool(os.getenv(self.embedding_api_key_env)),
                'reranker_configured': bool(self.reranker_url) if self.retrieval_backend == 'vllm' else bool(os.getenv(self.reranker_api_key_env))}
