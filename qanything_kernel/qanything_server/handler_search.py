from qanything_kernel.core.local_file import LocalFile
# from qanything_kernel.core.local_doc_search_cpu import LocalDocSearch
# from qanything_kernel.core.local_doc_search_npu import LocalDocSearch
from qanything_kernel.core.local_doc_search import LocalDocSearch
from qanything_kernel.utils.general_utils import *
from qanything_kernel.utils.custom_log import debug_logger, qa_logger
from sanic.response import ResponseStream
from sanic.response import json as sanic_json
from sanic.response import text as sanic_text
from sanic import request
import uuid
import json
import asyncio
import urllib.parse
import re
from datetime import datetime
import os
import time
from tqdm import tqdm
from save_apicsv import *

__all__ = ["list_kbs","list_docs","new_knowledge_base","document_parser", "document_parser_embedding", "chunk_embedding", "delete_knowledge_base", 
           "get_files_statu", "question_rag_search", "document", "upload_faqs"]

INVALID_USER_ID = f"fail, Invalid user_id: . user_id 必须只含有字母，数字和下划线且字母开头"


async def new_knowledge_base(req: request):
    local_doc_search: LocalDocSearch = req.app.ctx.local_doc_search
    user_id = safe_get(req, 'user_id')
    if user_id is None:
        return sanic_json({"code": 2002, "msg": f'输入非法！request.json：{req.json}，请检查！'})
    is_valid = validate_user_id(user_id)
    if not is_valid:
        return sanic_json({"code": 2005, "msg": get_invalid_user_id_msg(user_id=user_id)})
    debug_logger.info("new_knowledge_base %s", user_id)
    kb_name = safe_get(req, 'kb_name')
    kb_id = safe_get(req, 'kb_id', 'KB'+uuid.uuid4().hex)
    
    not_exist_kb_ids = local_doc_search.milvus_summary.check_kb_exist(user_id, [kb_id])
    if not not_exist_kb_ids:
        msg = "invalid kb_id: {}, is exist".format(not_exist_kb_ids)
        return sanic_json({"code": 2003, "msg": msg})
    
    local_doc_search.create_milvus_collection(user_id, kb_id, kb_name)
    now = datetime.now()
    timestamp = now.strftime("%Y%m%d%H%M")
    return sanic_json({"code": 200, "msg": "success create knowledge base {}".format(kb_id),
                       "data": {"kb_id": kb_id, "kb_name": kb_name, "timestamp": timestamp}})


async def upload_weblink(req: request):
    local_doc_search: LocalDocSearch = req.app.ctx.local_doc_search
    user_id = safe_get(req, 'user_id')
    if user_id is None:
        return sanic_json({"code": 2002, "msg": f'输入非法！request.json：{req.json}，请检查！'})
    is_valid = validate_user_id(user_id)
    if not is_valid:
        return sanic_json({"code": 2005, "msg": get_invalid_user_id_msg(user_id=user_id)})
    debug_logger.info("upload_weblink %s", user_id)
    kb_id = safe_get(req, 'kb_id')
    url = safe_get(req, 'url')
    mode = safe_get(req, 'mode', default='soft')  # soft代表不上传同名文件，strong表示强制上传同名文件
    not_exist_kb_ids = local_doc_search.milvus_summary.check_kb_exist(user_id, [kb_id])
    if not_exist_kb_ids:
        msg = "invalid kb_id: {}, please check...".format(not_exist_kb_ids)
        return sanic_json({"code": 2001, "msg": msg, "data": [{}]})
    now = datetime.now()
    timestamp = now.strftime("%Y%m%d%H%M")
    exist_files = []
    if mode == 'soft':
        exist_files = local_doc_search.milvus_summary.check_file_exist_by_name(user_id, kb_id, [url])
    if exist_files:
        file_id, file_name, file_size, status = exist_files[0]
        msg = f'warning，当前的mode是soft，无法上传同名文件，如果想强制上传同名文件，请设置mode：strong'
        data = [{"file_id": file_id, "file_name": url, "status": status, "bytes": file_size, "timestamp": timestamp}]
    else:
        file_id, msg = local_doc_search.milvus_summary.add_file(user_id, kb_id, url, timestamp)
        local_file = LocalFile(user_id, kb_id, url, file_id, url, local_doc_search.embeddings, is_url=True)
        data = [{"file_id": file_id, "file_name": url, "status": "gray", "bytes": 0, "timestamp": timestamp}]
        asyncio.create_task(local_doc_search.insert_files_to_milvus(user_id, kb_id, [local_file]))
        msg = "success，后台正在飞速上传文件，请耐心等待"
    return sanic_json({"code": 200, "msg": msg, "data": data})



async def document_parser(req: request):
    t1 = time.time()
    local_doc_search: LocalDocSearch = req.app.ctx.local_doc_search
    user_id = safe_get(req, 'user_id')
    if user_id is None:
        return sanic_json({"code": 2002, "msg": f'输入非法！request.form: {req.form}，request.files: {req.files}请检查！'})
    is_valid = validate_user_id(user_id)
    if not is_valid:
        return sanic_json({"code": 2005, "msg": get_invalid_user_id_msg(user_id=user_id)})
    debug_logger.info("document_parser %s", user_id)

    file = req.files.get('file')
    debug_logger.info('ori name: %s', file.name)
    file_name = urllib.parse.unquote(file.name, encoding='UTF-8')
    debug_logger.info('decode name: %s', file_name)
    file_name = re.sub(r'[\uFF01-\uFF5E\u3000-\u303F]', '', file_name)
    file_name = file_name.replace("/", "_")
    debug_logger.info('cleaned name: %s', file_name)
    file_name = truncate_filename(file_name)
    debug_logger.info('truncated name: %s', file_name)

    
    # 随机生成kb_id和file_id便于创建实例，进行文件解析
    kb_id = 'KB' + uuid.uuid4().hex
    file_id = uuid.uuid4().hex
    local_file = LocalFile(user_id, kb_id, file, file_id, file_name, local_doc_search.embeddings)    
    data = {"file_name": file_name, "bytes": len(local_file.file_content)}
    docs = local_file.parser_file(local_doc_search.get_ocr_result)
    
    parser_documents = []
    for doc in docs:
        parser_documents.append({'metadata': doc.metadata, 'page_content': doc.page_content})
    
    debug_logger.info("document_parser: %s", parser_documents)
    return_result = {"code": 200, "msg": "document parser success",
                        "parser_documents": parser_documents}

    try:
        t2 = time.time()    
        date = datetime.now().strftime("%Y-%m-%d")
        save_api_call_to_csv(date, "document_parser", req.form, return_result, t2-t1)
        debug_logger.info(f"time cost:{t2-t1}")
    except Exception as e:
        debug_logger.warn(f"save api 失败，异常信息：{e}")
        
    return sanic_json(return_result)
    

async def document_parser_embedding(req: request):
    t1 = time.time()
    local_doc_search: LocalDocSearch = req.app.ctx.local_doc_search
    user_id = safe_get(req, 'user_id')
    if user_id is None:
        return sanic_json({"code": 2002, "msg": f'输入非法！request.form: {req.form}，request.files: {req.files}请检查！'})
    is_valid = validate_user_id(user_id)
    if not is_valid:
        return sanic_json({"code": 2005, "msg": get_invalid_user_id_msg(user_id=user_id)})
    debug_logger.info("document_parser_embedding %s", user_id)
    kb_id = safe_get(req, 'kb_id')
    file_ids = safe_get(req, 'file_ids')
    file_ids = file_ids.split(',') if file_ids else []
    mode = safe_get(req, 'mode', default='soft')  # soft代表不上传同名文件，strong表示强制上传同名文件
    debug_logger.info("mode: %s", mode)
    files = req.files.getlist('files')

    # 判断file_ids与files是否匹配，数量是否一致
    if len(file_ids) != len(files):
        msg = "file_ids与files数量不一致，请检查！"
        debug_logger.info("%s", msg)
        return sanic_json({"code": 2001, "msg": msg, "data": [{}]})

    # 如果知识库id不存在，则创建知识库
    not_exist_kb_ids = local_doc_search.milvus_summary.check_kb_exist(user_id, [kb_id])
    if not_exist_kb_ids:
        msg = "invalid kb_id: {}, new_knowledge_base...".format(not_exist_kb_ids)
        debug_logger.info("%s", msg)
        local_doc_search.create_milvus_collection(user_id, kb_id, kb_id)    # 名称就用kb_id

    data = []
    local_files = []
    file_names = []
    for file in files:
        debug_logger.info('ori name: %s', file.name)
        file_name = urllib.parse.unquote(file.name, encoding='UTF-8')
        debug_logger.info('decode name: %s', file_name)
        # 删除掉全角字符
        file_name = re.sub(r'[\uFF01-\uFF5E\u3000-\u303F]', '', file_name)
        file_name = file_name.replace("/", "_")
        debug_logger.info('cleaned name: %s', file_name)
        file_name = truncate_filename(file_name)
        file_names.append(file_name)

    exist_file_names = []
    if mode == 'soft':
        exist_files = local_doc_search.milvus_summary.check_file_exist_by_name(user_id, kb_id, file_names)
        exist_file_names = [f[1] for f in exist_files]
        debug_logger.info("exist_file_names: %s", exist_file_names)

    now = datetime.now()
    timestamp = now.strftime("%Y%m%d%H%M")

    for file, file_name, file_id in zip(files, file_names, file_ids):
        if file_name in exist_file_names:
            continue
        file_id, msg = local_doc_search.milvus_summary.add_fileid(user_id, kb_id, file_id, file_name, timestamp)
        debug_logger.info(f"{file_name}, {file_id}, {msg}")
        if (file_id is None):
            return sanic_json({"code": 2001, "msg": msg, "data": [{}]})
        local_file = LocalFile(user_id, kb_id, file, file_id, file_name, local_doc_search.embeddings)
        local_files.append(local_file)
        local_doc_search.milvus_summary.update_file_size(file_id, len(local_file.file_content))
        data.append(
            {"file_id": file_id, "file_name": file_name, "status": "gray", "bytes": len(local_file.file_content),
             "timestamp": timestamp})

    asyncio.create_task(local_doc_search.insert_files_to_milvus(user_id, kb_id, local_files))
    if exist_file_names:
        msg = f'warning，当前的mode是soft，无法上传同名文件{exist_file_names}，如果想强制上传同名文件，请设置mode：strong'
    else:
        msg = "success，后台正在飞速上传文件，请耐心等待"
    
    return_result = {"code": 200, "msg": msg, "data": data}

    try:
        t2 = time.time()    
        date = datetime.now().strftime("%Y-%m-%d")
        save_api_call_to_csv(date, "document_parser_embedding", req.form, return_result, t2-t1)
        debug_logger.info(f"time cost:{t2-t1}")
    except Exception as e:
        debug_logger.warn(f"save api 失败，异常信息：{e}")
    return sanic_json(return_result)
    

async def chunk_embedding(req: request):
    t1 = time.time()
    local_doc_search: LocalDocSearch = req.app.ctx.local_doc_search
    user_id = safe_get(req, 'user_id')
    if user_id is None:
        return sanic_json({"code": 2002, "msg": f'输入非法！request.form: {req.form}，request.files: {req.files}请检查！'})
    is_valid = validate_user_id(user_id)
    if not is_valid:
        return sanic_json({"code": 2005, "msg": get_invalid_user_id_msg(user_id=user_id)})
    debug_logger.info("document_parser_embedding %s", user_id)
    kb_id = safe_get(req, 'kb_id')
    file_id = safe_get(req, 'file_id', uuid.uuid4().hex)
    file_name = safe_get(req, 'file_name', file_id+".txt")
    chunk_datas = safe_get(req, 'chunk_datas')
    if not isinstance(chunk_datas, list):
        return sanic_json({"code": 2003, "msg": f'输入chunk格式非法！请检查！'})
    
    
    # 如果知识库id不存在，则创建知识库
    not_exist_kb_ids = local_doc_search.milvus_summary.check_kb_exist(user_id, [kb_id])
    if not_exist_kb_ids:
        msg = "invalid kb_id: {}, new_knowledge_base...".format(not_exist_kb_ids)
        debug_logger.info("%s", msg)
        local_doc_search.create_milvus_collection(user_id, kb_id, kb_id)    # 名称就用kb_id

   
    now = datetime.now()
    timestamp = now.strftime("%Y%m%d%H%M")
    
        
    file_id, msg = local_doc_search.milvus_summary.add_fileid(user_id, kb_id, file_id, file_name, timestamp)
    debug_logger.info(f"{file_name}, {file_id}, {msg}")
    if file_id is None:
        return sanic_json({"code": 2004, "msg": msg})
    
    local_file = LocalFile(user_id, kb_id, chunk_datas, file_id, file_name, local_doc_search.embeddings)
    local_doc_search.milvus_summary.update_file_size(file_id, len(local_file.file_content))
    data ={"file_id": file_id, "file_name": file_name, "status": "gray", "bytes": len(local_file.file_content),
            "timestamp": timestamp}

    asyncio.create_task(local_doc_search.insert_files_to_milvus(user_id, kb_id, [local_file]))
    msg = "success，后台正在飞速上传文件，请耐心等待"
    
    return_result = {"code": 200, "msg": msg, "data": data}

    try:
        t2 = time.time()    
        date = datetime.now().strftime("%Y-%m-%d")
        save_api_call_to_csv(date, "chunk_embedding", req.form, return_result, t2-t1)
        debug_logger.info(f"time cost:{t2-t1}")
    except Exception as e:
        debug_logger.warn(f"save api 失败，异常信息：{e}")
    return sanic_json(return_result)


async def list_kbs(req: request):
    local_doc_search: LocalDocSearch = req.app.ctx.local_doc_search
    user_id = safe_get(req, 'user_id')
    if user_id is None:
        return sanic_json({"code": 2002, "msg": f'输入非法！request.json：{req.json}，请检查！'})
    is_valid = validate_user_id(user_id)
    if not is_valid:
        return sanic_json({"code": 2005, "msg": get_invalid_user_id_msg(user_id=user_id)})
    debug_logger.info("list_kbs %s", user_id)
    kb_infos = local_doc_search.milvus_summary.get_knowledge_bases(user_id)
    data = []
    for kb in kb_infos:
        data.append({"kb_id": kb[0], "kb_name": kb[1]})
    debug_logger.info("all kb infos: {}".format(data))
    return sanic_json({"code": 200, "data": data})


async def list_docs(req: request):
    local_doc_search: LocalDocSearch = req.app.ctx.local_doc_search
    user_id = safe_get(req, 'user_id')
    if user_id is None:
        return sanic_json({"code": 2002, "msg": f'输入非法！request.json：{req.json}，请检查！'})
    is_valid = validate_user_id(user_id)
    if not is_valid:
        return sanic_json({"code": 2005, "msg": get_invalid_user_id_msg(user_id=user_id)})
    debug_logger.info("list_docs %s", user_id)
    kb_id = safe_get(req, 'kb_id')
    debug_logger.info("kb_id: {}".format(kb_id))
    data = []
    file_infos = local_doc_search.milvus_summary.get_files(user_id, kb_id)
    status_count = {}
    msg_map = {'gray': "正在上传中，请耐心等待",
               'red': "split或embedding失败，请检查文件类型，仅支持[md,txt,pdf,jpg,png,jpeg,docx,xlsx,pptx,eml,csv]",
               'yellow': "milvus插入失败，请稍后再试", 'green': "上传成功"}
    for file_info in file_infos:
        status = file_info[2]
        if status not in status_count:
            status_count[status] = 1
        else:
            status_count[status] += 1
        data.append({"file_id": file_info[0], "file_name": file_info[1], "status": file_info[2], "bytes": file_info[3],
                     "content_length": file_info[4], "timestamp": file_info[5], "msg": msg_map[file_info[2]]})

    return sanic_json({"code": 200, "msg": "success", "data": {'total': status_count, 'details': data}})


async def get_files_statu(req: request):
    local_doc_search: LocalDocSearch = req.app.ctx.local_doc_search
    user_id = safe_get(req, 'user_id')
    if user_id is None:
        return sanic_json({"code": 2002, "msg": f'输入非法！request.json：{req.json}，请检查！'})
    is_valid = validate_user_id(user_id)
    if not is_valid:
        return sanic_json({"code": 2005, "msg": get_invalid_user_id_msg(user_id=user_id)})
    debug_logger.info("list_docs %s", user_id)
    kb_id = safe_get(req, 'kb_id')
    debug_logger.info("kb_id: {}".format(kb_id))
    file_ids = safe_get(req, 'file_ids', [])
    debug_logger.info("file_ids: {}".format(file_ids))

    # 如果没有输入file_ids,则返回当前知识库下的所有文件信息
    data = []
    if len(file_ids) == 0:
        file_infos = local_doc_search.milvus_summary.get_files(user_id, kb_id)
    else:
        file_infos = local_doc_search.milvus_summary.get_files_info(user_id, kb_id, file_ids)
    status_count = {}
    msg_map = {'gray': "正在上传中，请耐心等待",
               'red': "split或embedding失败，请检查文件类型，仅支持[md,txt,pdf,jpg,png,jpeg,docx,xlsx,pptx,eml,csv]",
               'yellow': "milvus插入失败，请稍后再试", 
               'green': "上传成功"}
    for file_info in file_infos:
        status = file_info[2]
        if status not in status_count:
            status_count[status] = 1
        else:
            status_count[status] += 1
        data.append({"file_id": file_info[0], "file_name": file_info[1], "status": file_info[2], "bytes": file_info[3],
                     "content_length": file_info[4], "timestamp": file_info[5], "msg": msg_map[file_info[2]]})

    return sanic_json({"code": 200, "msg": "success", "data": {'total': status_count, 'details': data}})


async def delete_knowledge_base(req: request):
    local_doc_search: LocalDocSearch = req.app.ctx.local_doc_search
    user_id = safe_get(req, 'user_id')
    if user_id is None:
        return sanic_json({"code": 2002, "msg": f'输入非法！request.json：{req.json}，请检查！'})
    is_valid = validate_user_id(user_id)
    if not is_valid:
        return sanic_json({"code": 2005, "msg": get_invalid_user_id_msg(user_id=user_id)})
    debug_logger.info("delete_knowledge_base %s", user_id)
    kb_id = safe_get(req, 'kb_id')
    file_ids = safe_get(req, "file_ids", [])
    
    if len(file_ids) > 0:
        not_exist_kb_ids = local_doc_search.milvus_summary.check_kb_exist(user_id, [kb_id])
        if not_exist_kb_ids:
            return sanic_json({"code": 2003, "msg": "fail, knowledge Base {} not found".format(not_exist_kb_ids[0])})
        valid_file_infos = local_doc_search.milvus_summary.check_file_exist(user_id, kb_id, file_ids)
        if len(valid_file_infos) == 0:
            return sanic_json({"code": 2004, "msg": "fail, files {} not found".format(file_ids)})
        milvus_kb = local_doc_search.match_milvus_kb(user_id, [kb_id])
        milvus_kb.delete_files(file_ids)
        # 删除数据库中的记录
        local_doc_search.milvus_summary.delete_files(kb_id, file_ids)
        return sanic_json({"code": 200, "msg": "documents {} delete success".format(file_ids)})
    
    else:    
        not_exist_kb_ids = local_doc_search.milvus_summary.check_kb_exist(user_id, [kb_id])
        if not_exist_kb_ids:
            return sanic_json({"code": 2003, "msg": "fail, knowledge Base {} not found".format(not_exist_kb_ids)})

        milvus = local_doc_search.match_milvus_kb(user_id, [kb_id])
        milvus.delete_partition(kb_id)
        local_doc_search.milvus_summary.delete_knowledge_base(user_id, [kb_id])
        return sanic_json({"code": 200, "msg": "Knowledge Base {} delete success".format(kb_id)})


async def rename_knowledge_base(req: request):
    local_doc_search: LocalDocSearch = req.app.ctx.local_doc_search
    user_id = safe_get(req, 'user_id')
    if user_id is None:
        return sanic_json({"code": 2002, "msg": f'输入非法！request.json：{req.json}，请检查！'})
    is_valid = validate_user_id(user_id)
    if not is_valid:
        return sanic_json({"code": 2005, "msg": get_invalid_user_id_msg(user_id=user_id)})
    debug_logger.info("rename_knowledge_base %s", user_id)
    kb_id = safe_get(req, 'kb_id')
    new_kb_name = safe_get(req, 'new_kb_name')
    not_exist_kb_ids = local_doc_search.milvus_summary.check_kb_exist(user_id, [kb_id])
    if not_exist_kb_ids:
        return sanic_json({"code": 2003, "msg": "fail, knowledge Base {} not found".format(not_exist_kb_ids[0])})
    local_doc_search.milvus_summary.rename_knowledge_base(user_id, kb_id, new_kb_name)
    return sanic_json({"code": 200, "msg": "Knowledge Base {} rename success".format(kb_id)})


async def delete_docs(req: request):
    local_doc_search: LocalDocSearch = req.app.ctx.local_doc_search
    user_id = safe_get(req, 'user_id')
    if user_id is None:
        return sanic_json({"code": 2002, "msg": f'输入非法！request.json：{req.json}，请检查！'})
    is_valid = validate_user_id(user_id)
    if not is_valid:
        return sanic_json({"code": 2005, "msg": get_invalid_user_id_msg(user_id=user_id)})
    debug_logger.info("delete_docs %s", user_id)
    kb_id = safe_get(req, 'kb_id')
    file_ids = safe_get(req, "file_ids")
    not_exist_kb_ids = local_doc_search.milvus_summary.check_kb_exist(user_id, [kb_id])
    if not_exist_kb_ids:
        return sanic_json({"code": 2003, "msg": "fail, knowledge Base {} not found".format(not_exist_kb_ids[0])})
    valid_file_infos = local_doc_search.milvus_summary.check_file_exist(user_id, kb_id, file_ids)
    if len(valid_file_infos) == 0:
        return sanic_json({"code": 2004, "msg": "fail, files {} not found".format(file_ids)})
    milvus_kb = local_doc_search.match_milvus_kb(user_id, [kb_id])
    milvus_kb.delete_files(file_ids)
    # 删除数据库中的记录
    local_doc_search.milvus_summary.delete_files(kb_id, file_ids)
    return sanic_json({"code": 200, "msg": "documents {} delete success".format(file_ids)})


async def get_total_status(req: request):
    local_doc_search: LocalDocSearch = req.app.ctx.local_doc_search
    user_id = safe_get(req, 'user_id')
    if user_id is None:
        return sanic_json({"code": 2002, "msg": f'输入非法！request.json：{req.json}，请检查！'})
    is_valid = validate_user_id(user_id)
    if not is_valid:
        return sanic_json({"code": 2005, "msg": get_invalid_user_id_msg(user_id=user_id)})
    debug_logger.info('get_total_status %s', user_id)
    if not user_id:
        users = local_doc_search.milvus_summary.get_users()
        users = [user[0] for user in users]
    else:
        users = [user_id]
    res = {}
    for user in users:
        res[user] = {}
        kbs = local_doc_search.milvus_summary.get_knowledge_bases(user)
        for kb_id, kb_name in kbs:
            gray_file_infos = local_doc_search.milvus_summary.get_file_by_status([kb_id], 'gray')
            red_file_infos = local_doc_search.milvus_summary.get_file_by_status([kb_id], 'red')
            yellow_file_infos = local_doc_search.milvus_summary.get_file_by_status([kb_id], 'yellow')
            green_file_infos = local_doc_search.milvus_summary.get_file_by_status([kb_id], 'green')
            res[user][kb_name + kb_id] = {'green': len(green_file_infos), 'yellow': len(yellow_file_infos),
                                          'red': len(red_file_infos),
                                          'gray': len(gray_file_infos)}

    return sanic_json({"code": 200, "status": res})


async def clean_files_by_status(req: request):
    local_doc_search: LocalDocSearch = req.app.ctx.local_doc_search
    user_id = safe_get(req, 'user_id')
    if user_id is None:
        return sanic_json({"code": 2002, "msg": f'输入非法！request.json：{req.json}，请检查！'})
    is_valid = validate_user_id(user_id)
    if not is_valid:
        return sanic_json({"code": 2005, "msg": get_invalid_user_id_msg(user_id=user_id)})
    debug_logger.info('clean_files_by_status %s', user_id)
    status = safe_get(req, 'status', default='gray')
    kb_ids = safe_get(req, 'kb_ids')
    if not kb_ids:
        kbs = local_doc_search.milvus_summary.get_knowledge_bases(user_id)
        kb_ids = [kb[0] for kb in kbs]
    else:
        not_exist_kb_ids = local_doc_search.milvus_summary.check_kb_exist(user_id, kb_ids)
        if not_exist_kb_ids:
            return sanic_json({"code": 2003, "msg": "fail, knowledge Base {} not found".format(not_exist_kb_ids)})

    gray_file_infos = local_doc_search.milvus_summary.get_file_by_status(kb_ids, status)
    gray_file_ids = [f[0] for f in gray_file_infos]
    gray_file_names = [f[1] for f in gray_file_infos]
    debug_logger.info(f'{status} files number: {len(gray_file_names)}')
    # 删除milvus中的file
    if gray_file_ids:
        milvus_kb = local_doc_search.match_milvus_kb(user_id, kb_ids)
        milvus_kb.delete_files(gray_file_ids)
        for kb_id in kb_ids:
            local_doc_search.milvus_summary.delete_files(kb_id, gray_file_ids)
    return sanic_json({"code": 200, "msg": f"delete {status} files success", "data": gray_file_names})


async def question_rag_search(req: request):
    t1 = time.time()
    local_doc_search: LocalDocSearch = req.app.ctx.local_doc_search
    user_id = safe_get(req, 'user_id')
    if user_id is None:
        return sanic_json({"code": 2002, "msg": f'输入非法！request.json：{req.json}，请检查！'})
    is_valid = validate_user_id(user_id)
    if not is_valid:
        return sanic_json({"code": 2005, "msg": get_invalid_user_id_msg(user_id=user_id)})
    debug_logger.info('local_doc_chat %s', user_id)
    kb_ids = safe_get(req, 'kb_ids')
    question = safe_get(req, 'question')
    rerank = safe_get(req, 'rerank', default=True)
    
    debug_logger.info('rerank %s', rerank)
    debug_logger.info("question: %s", question)
    debug_logger.info("kb_ids: %s", kb_ids)
    debug_logger.info("user_id: %s", user_id)

    not_exist_kb_ids = local_doc_search.milvus_summary.check_kb_exist(user_id, kb_ids)
    if not_exist_kb_ids:
        return sanic_json({"code": 2003, "msg": "fail, knowledge Base {} not found".format(not_exist_kb_ids)})

    file_infos = []
    milvus_kb = local_doc_search.match_milvus_kb(user_id, kb_ids)
    for kb_id in kb_ids:
        file_infos.extend(local_doc_search.milvus_summary.get_files(user_id, kb_id))
    valid_files = [fi for fi in file_infos if fi[2] == 'green']
    if len(valid_files) == 0:
        return sanic_json({"code": 200, "msg": "当前知识库为空，请上传文件或等待文件解析完毕", "question": question,
                           "response": "All knowledge bases {} are empty or haven't green file, please upload files".format(
                               kb_ids), "history": history, "source_documents": [{}]})
    else:
        retrieval_documents = local_doc_search.get_knowledge_based(
                query=question, milvus_kb=milvus_kb, rerank=rerank)
        
        chat_data = {'user_id': user_id, 'kb_ids': kb_ids, 'query': question, 
                        'retrieval_documents': retrieval_documents}
        qa_logger.info("question_rag_search: %s", chat_data)
        debug_logger.info("question_rag_search: %s", chat_data)

        source_documents =[]
        for doc in retrieval_documents:
            source_documents.append({'metadata': doc.metadata, 'page_content': doc.page_content})
        return_result = {"code": 200, "msg": "success chat", "question": question, 
                            "retrieval_documents": source_documents}

        try:
            t2 = time.time()    
            date = datetime.now().strftime("%Y-%m-%d")
            save_api_call_to_csv(date, "question_rag_search", req.form, return_result, t2-t1)
            debug_logger.info(f"question_rag_search time cost:{t2-t1}")
        except Exception as e:
            debug_logger.warn(f"save api 失败，异常信息：{e}")
        return sanic_json(return_result)


async def document(req: request):
    description = """
# QAnything 介绍
[戳我看视频>>>>>【有道QAnything介绍视频.mp4】](https://docs.popo.netease.com/docs/7e512e48fcb645adadddcf3107c97e7c)

**QAnything** (**Q**uestion and **A**nswer based on **Anything**) 是支持任意格式的本地知识库问答系统。

您的任何格式的本地文件都可以往里扔，即可获得准确、快速、靠谱的问答体验。

**目前已支持格式:**
* PDF
* Word(doc/docx)
* PPT
* TXT
* 图片
* 网页链接
* ...更多格式，敬请期待


## API 接口说明
    {
        "api": "/api/docs"
        "name": "接口文档",
        "description": "获取接口文档",
    },
    {
        "api": "/api/qanything/new_knowledge_base"
        "name": "新建知识库",
        "description": "新建一个知识库，需要提供知识库的名称和描述",
    },
    {
        "api": "/api/qanything/delete_knowledge_base"
        "name": "删除知识库",
        "description": "删除一个知识库，需要提供知识库的ID",
    },
    {
        "api": "/api/qanything/document_parser"
        "name": "解析文件",
        "description": "解析一个文件，需要提供文件的URL和知识库的ID",
    },
    {
        "api": "/api/qanything/document_parser_embedding"
        "name": "解析文件并保存",
        "description": "解析一个文件并将其保存到知识库中，需要提供文件的URL和知识库的ID",
    },
    {
        "api": "/api/qanything/delete_files"
        "name": "删除文件",
        "description": "删除一个知识库下的文件，需要提供文件的ID和知识库的ID",
    },
    {
        "api": "/api/qanything/question_rag_search"
        "name": "问答接口",
        "description": "问答接口，需要提供知识库的ID、用户问题和history（支持多轮对话）",
    },
    {
        "api": "/api/qanything/list_knowledge_base"
        "name": "知识库列表",
        "description": "获取所有知识库的列表",
    },
    {
        "api": "/api/qanything/list_files"
        "name": "文件列表",
        "description": "获取指定知识库下的文件列表，需要提供知识库的ID",
    },
    {
        "api": "/api/qanything/get_total_status"
        "name": "获取所有知识库状态",
        "description": "获取所有知识库的状态信息",
    },
    {
        "api": "/api/qanything/upload_faqs"
        "name": "上传FAQ",
        "description": "上传一个FAQ文件到知识库，需要提供知识库的ID和FAQ文件的URL",
    }

"""

    return sanic_text(description)
    
    # html_str = markdown.markdown(description)    
    # return sanic_text(html_str)


async def upload_faqs(req: request):
    local_doc_search: LocalDocSearch = req.app.ctx.local_doc_search
    user_id = safe_get(req, 'user_id')
    if user_id is None:
        return sanic_json({"code": 2002, "msg": f'输入非法！request.json：{req.json}，请检查！'})
    is_valid = validate_user_id(user_id)
    if not is_valid:
        return sanic_json({"code": 2005, "msg": get_invalid_user_id_msg(user_id=user_id)})
    debug_logger.info("upload_faqs %s", user_id)
    kb_id = safe_get(req, 'kb_id')
    debug_logger.info("kb_id %s", kb_id)
    faqs = safe_get(req, 'faqs')
    file_status = {}
    if faqs is None:
        files = req.files.getlist('files')
        faqs = []
        for file in files:
            debug_logger.info('ori name: %s', file.name)
            file_name = urllib.parse.unquote(file.name, encoding='UTF-8')
            debug_logger.info('decode name: %s', file_name)
            # 删除掉全角字符
            file_name = re.sub(r'[\uFF01-\uFF5E\u3000-\u303F]', '', file_name)
            file_name = file_name.replace("/", "_")
            debug_logger.info('cleaned name: %s', file_name)
            file_name = truncate_filename(file_name)
            file_faqs = check_and_transform_excel(file.body)
            if isinstance(file_faqs, str):
                file_status[file_name] = file_faqs
            else:
                faqs.extend(file_faqs)
                file_status[file_name] = "success"

    if len(faqs) > 1000:
        return sanic_json({"code": 2002, "msg": f"fail, faqs too many, The maximum length of each request is 1000."})

    not_exist_kb_ids = local_doc_search.milvus_summary.check_kb_exist(user_id, [kb_id])
    if not_exist_kb_ids:
        msg = "invalid kb_id: {}, please check...".format(not_exist_kb_ids)
        return sanic_json({"code": 2001, "msg": msg})

    data = []
    now = datetime.now()
    local_files = []
    timestamp = now.strftime("%Y%m%d%H%M")
    debug_logger.info(f"start insert {len(faqs)} faqs to mysql, user_id: {user_id}, kb_id: {kb_id}")
    debug_logger.info(f"faqs: {faqs}")
    exist_questions = []
    for faq in tqdm(faqs):
        ques = faq['question']
        if ques not in exist_questions:
            exist_questions.append(ques)
        else:
            debug_logger.info(f"question {ques} already exists, skip it")
            continue
        if len(ques) > 512 or len(faq['answer']) > 2048:
            return sanic_json(
                {"code": 2003, "msg": f"fail, faq too long, max length of question is 512, answer is 2048."})
        content_length = len(ques) + len(faq['answer'])
        file_name = f"FAQ_{ques}.faq"
        file_name = file_name.replace("/", "_").replace(":", "_")  # 文件名中的/和：会导致写入时出错
        file_name = simplify_filename(file_name)
        file_id, msg = local_doc_search.milvus_summary.add_file(user_id, kb_id, file_name, timestamp, status='green')
        debug_logger.info('file_name:{}'.format(file_name))
        debug_logger.info('file_id:{}'.format(file_id))
        local_file = LocalFile(user_id, kb_id, faq, file_id, file_name, local_doc_search.embeddings)
        local_doc_search.milvus_summary.update_file_path(file_id, local_file.file_path)
        local_files.append(local_file)
        local_doc_search.milvus_summary.add_faq(file_id, user_id, kb_id, ques, faq['answer'], faq.get("nos_keys", ""))
        # debug_logger.info(f"{file_name}, {file_id}, {msg}, {faq}")
        data.append(
            {"file_id": file_id, "file_name": file_name, "status": "gray", "length": content_length,
             "timestamp": timestamp})
    debug_logger.info(f"end insert {len(faqs)} faqs to mysql, user_id: {user_id}, kb_id: {kb_id}")

    asyncio.create_task(local_doc_search.insert_files_to_milvus(user_id, kb_id, local_files))

    msg = "success，后台正在飞速上传文件，请耐心等待"
    return sanic_json({"code": 200, "msg": msg, "file_status": file_status, "data": data})
