import {useResource,qs} from './api';
import type {PageProps,Repository} from './types';
import {ErrorBox,names} from './ui';

export default function RepositorySelect({workspace,session,jobTick,onSession,onError}:PageProps){
  const data=useResource<{items:Repository[]}>('/repositories?'+qs({workspace_id:workspace.id}),jobTick);
  return <div className="filters"><label>关联仓库 <select aria-label="当前 Repository" value={session?.repository_id||''} onChange={e=>onSession('code',e.target.value).catch(e=>onError(e.message))}><option value="">不关联仓库（仅使用对话记忆）</option>{data.data?.items.map(repo=><option key={repo.id} value={repo.id}>{repo.name} · {names[repo.status]||repo.status}</option>)}</select></label><a className="secondary-button" href="#repositories">管理仓库与代码结构</a>{data.error&&<ErrorBox error={data.error} onRetry={data.reload}/>}</div>
}
