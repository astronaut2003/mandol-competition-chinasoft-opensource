from typing import Literal
from pydantic import BaseModel, ConfigDict, Field, model_validator

Mode = Literal['textual', 'code', 'multimodal']


class Strict(BaseModel):
    model_config = ConfigDict(extra='forbid', strict=True)


class Message(Strict):
    role: Literal['user', 'assistant'] = 'user'
    content: str = Field(default='', max_length=16000)
    media_ids: list[str] = Field(default_factory=list, max_length=4)

    @model_validator(mode='after')
    def nonempty(self):
        if not self.content.strip() and not self.media_ids:
            raise ValueError('Text or an image is required')
        return self


class AddMemory(Strict):
    workspace_id: str = Field(min_length=1, max_length=80)
    session_id: str = Field(min_length=1, max_length=80)
    request_id: str = Field(min_length=1, max_length=100)
    mode: Mode = 'textual'
    repository_id: str | None = Field(default=None, max_length=80)
    messages: list[Message] = Field(min_length=1, max_length=32)
    attachment_ids: list[str] = Field(default_factory=list,max_length=4)

    @model_validator(mode='after')
    def budget(self):
        if sum(len(m.content) for m in self.messages) > 64000:
            raise ValueError('Input exceeds 64,000 characters')
        return self


class Chat(Strict):
    workspace_id: str
    session_id: str
    request_id: str = Field(min_length=1, max_length=100)
    mode: Literal['auto', 'textual', 'code', 'multimodal'] = 'auto'
    content: str = Field(min_length=1, max_length=8000)
    media_ids: list[str] = Field(default_factory=list, max_length=4)
    attachment_ids: list[str] = Field(default_factory=list,max_length=4)
    repository_id: str | None = Field(default=None, max_length=80)
    retrieval_scope: Literal['current','workspace','conversation','conversations'] = 'conversation'
    reasoning_mode: Literal['fast','standard','deep'] = 'fast'
    web_search: bool = False
    web_query: str = Field(default='',max_length=128,description='Optional legacy search override; omitted by the unified Composer')


class Workspace(Strict):
    name: str = Field(min_length=1, max_length=60)


class ClearMemories(Strict):
    confirmation: str = Field(max_length=40)
    request_id: str = Field(min_length=1, max_length=100)


class Session(Strict):
    workspace_id: str
    mode: Mode = 'textual'
    title: str = Field(default='新会话', min_length=1, max_length=80)
    repository_id: str | None = Field(default=None,max_length=80)
    unified: bool = True

class SessionUpdate(Strict):
    unified: bool | None = None
    title: str | None = Field(default=None,min_length=1,max_length=80)
    archived: bool | None = None
    repository_id: str | None = Field(default=None,max_length=80)

class Credentials(Strict):
    username: str = Field(min_length=1,max_length=64)
    password: str = Field(min_length=1,max_length=128)

class Registration(Credentials):
    display_name: str = Field(default='',max_length=60)
    password_confirmation: str | None = Field(default=None,max_length=128)

    @model_validator(mode='after')
    def matching_password(self):
        if self.password_confirmation is not None and self.password_confirmation!=self.password:raise ValueError('两次输入的密码不一致')
        return self


class AccountChange(Strict):
    status: Literal['active','disabled','pending'] | None = None
    role: Literal['admin','user'] | None = None
    revoke_sessions: bool = False
    confirm: bool = False
    current_password: str = Field(default='',max_length=128)


class QuotaUpdate(Strict):
    limits: dict[str,int] | None = None


class RemoteRepositoryRegistration(Strict):
    workspace_id: str = Field(min_length=1,max_length=80)
    url: str = Field(min_length=1,max_length=500)
    ref: str = Field(default='',max_length=128)
    name: str = Field(default='',max_length=80)
    request_id: str = Field(min_length=1,max_length=100)


class RepositoryRegistration(Strict):
    workspace_id: str = Field(min_length=1,max_length=80)
    root: str = Field(min_length=1,max_length=1000)
    name: str = Field(default='代码仓库',min_length=1,max_length=80)
    request_id: str = Field(min_length=1,max_length=100)


class RepositoryScan(Strict):
    request_id: str = Field(min_length=1,max_length=100)


class Revision(Strict):
    request_id: str = Field(min_length=1, max_length=100)
    expected_version: int = Field(ge=1)
    content: str = Field(min_length=1, max_length=16000)


class Retry(Strict):
    allow_unknown: bool = False
