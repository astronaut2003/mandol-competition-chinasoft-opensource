import {chromium,expect} from '@playwright/test';
import fs from 'node:fs';
import path from 'node:path';
const root=path.resolve(import.meta.dirname,'../..');
const output=path.join(root,'.release-private/browser');fs.mkdirSync(output,{recursive:true,mode:0o700});
const passwordFile=process.env.MANDOL_E2E_PASSWORD_FILE;
if(!passwordFile||!process.env.MANDOL_E2E_USERNAME)throw Error('Set MANDOL_E2E_USERNAME and a private MANDOL_E2E_PASSWORD_FILE');
if(fs.statSync(passwordFile).mode&0o077)throw Error('Password file must be 0600');
const browser=await chromium.launch({headless:true,args:['--disable-dev-shm-usage']});
const context=await browser.newContext({viewport:{width:1440,height:900}});const page=await context.newPage();const errors=[];page.on('pageerror',e=>errors.push(e.name));
const results=[];let failed=false;
async function check(name,fn){try{await fn();results.push({name,status:'PASS'});}catch(e){failed=true;results.push({name,status:'FAIL',error:e.message.slice(0,250)});}}
try{
 await page.goto(process.env.MANDOL_E2E_URL||'http://127.0.0.1:20600');
 await page.getByLabel('账号',{exact:true}).fill(process.env.MANDOL_E2E_USERNAME);await page.getByLabel('密码',{exact:true}).fill(fs.readFileSync(passwordFile,'utf8').trim());await page.getByRole('button',{name:'登录',exact:true}).click();
 await expect(page.locator('.sidebar')).toBeVisible();
 await check('authenticated real frontend',async()=>{await expect(page.getByLabel('消息或记忆内容')).toBeVisible();});
 await page.getByTitle('新建会话',{exact:true}).click();await expect(page.getByLabel('消息或记忆内容')).toBeEnabled();
 await check('unified composer and empty conversation',async()=>{await expect(page.locator('.composer textarea')).toHaveCount(1);await expect(page.locator('input[placeholder*="公开搜索"]')).toHaveCount(0);await page.screenshot({path:path.join(output,'desktop.png')});});
 await check('web toggle and real reasoning selector',async()=>{const b=page.getByRole('button',{name:'联网搜索',exact:true});if(await b.isEnabled()){await b.click();await expect(b).toHaveAttribute('aria-pressed','true');await b.click();}await page.locator('.composer select').selectOption('fast');});
 await check('safe code and image attachment upload',async()=>{await page.getByLabel('上传附件',{exact:true}).setInputFiles([path.join(root,'examples/tiny-repository/calculator.py'),path.join(root,'demo/status_board.png')]);await expect(page.getByLabel('移除附件 calculator.py',{exact:true})).toBeVisible({timeout:20000});await expect(page.getByLabel('移除附件 status_board.png',{exact:true})).toBeVisible({timeout:20000});await page.getByLabel('消息或记忆内容').fill('Synthetic browser fixture: explain this code.');await page.screenshot({path:path.join(output,'attachments.png')});await page.getByLabel('移除附件 calculator.py',{exact:true}).click();await page.getByLabel('移除附件 status_board.png',{exact:true}).click();});
 await check('durable save through composer',async()=>{await page.getByLabel('消息或记忆内容').fill('Synthetic browser fixture: the notebook cover is blue.');const response=page.waitForResponse(r=>r.url().endsWith('/api/memories')&&r.request().method()==='POST');await page.getByRole('button',{name:'仅保存',exact:true}).click();expect((await response).status()).toBe(202);await expect(page.getByLabel('消息或记忆内容')).toHaveValue('');});
 for(const [width,height] of [[1920,1080],[1280,720],[768,1024],[390,844]])await check('composer viewport '+width,async()=>{await page.setViewportSize({width,height});await page.screenshot({path:path.join(output,`viewport-${width}.png`)});const b=await page.locator('.composer').boundingBox();expect(b.x).toBeGreaterThanOrEqual(-1);expect(b.x+b.width).toBeLessThanOrEqual(width+1);await expect(page.getByRole('button',{name:'发送消息',exact:true})).toBeVisible();});
 await page.setViewportSize({width:1440,height:900});
 await check('memory schema and current/history graph navigation',async()=>{await page.goto((process.env.MANDOL_E2E_URL||'http://127.0.0.1:20600')+'/#memories');await expect(page.getByRole('button',{name:'记忆模式',exact:true})).toHaveClass(/active/);await page.screenshot({path:path.join(output,'schema.png')});await page.getByRole('button',{name:'知识关系图',exact:true}).click();await page.getByLabel('知识状态',{exact:true}).selectOption('history');await page.getByLabel('知识状态',{exact:true}).selectOption('current');await page.screenshot({path:path.join(output,'graph.png')});});
 await check('dark theme',async()=>{await page.getByLabel('切换主题',{exact:true}).click();await page.screenshot({path:path.join(output,'dark.png')});});
 await check('refresh and re-login persistence',async()=>{await page.reload();await expect(page.locator('.sidebar')).toBeVisible();await page.getByLabel('退出登录',{exact:true}).click();await expect(page.getByLabel('账号',{exact:true})).toBeVisible();await page.getByLabel('账号',{exact:true}).fill(process.env.MANDOL_E2E_USERNAME);await page.getByLabel('密码',{exact:true}).fill(fs.readFileSync(passwordFile,'utf8').trim());await page.getByRole('button',{name:'登录',exact:true}).click();await expect(page.locator('.sidebar')).toBeVisible();});
 await check('no browser JavaScript exceptions',async()=>expect(errors).toEqual([]));
}finally{fs.writeFileSync(path.join(output,'results.json'),JSON.stringify(results,null,2));await browser.close();}
console.log(JSON.stringify(results,null,2));if(failed)process.exitCode=1;
