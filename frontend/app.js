// RAG 知识库问答前端
const API_BASE = window.location.origin + '/api';

// 状态管理
let currentSessionId = null;
let uploadedFiles = [];
let selectedFiles = [];
let processingStatus = 'idle';
let fileToDelete = null;
let isRequestInProgress = false;  // 请求锁，防止重复提交

// DOM 元素
const mainView = document.getElementById('main-view');
const chatView = document.getElementById('chat-view');
const questionInput = document.getElementById('question-input');
const chatInput = document.getElementById('chat-input');
const sendBtn = document.getElementById('send-btn');
const chatSendBtn = document.getElementById('chat-send-btn');
const messageList = document.getElementById('message-list');
const loading = document.getElementById('loading');
const docList = document.getElementById('doc-list');
const docCount = document.getElementById('doc-count');
const processingStatusEl = document.getElementById('processing-status');
const statusTextEl = document.getElementById('status-text');
const deleteModal = document.getElementById('delete-modal');
const deleteFileName = document.getElementById('delete-file-name');

// 初始化
document.addEventListener('DOMContentLoaded', () => {
    initEventListeners();
    loadDocuments();
});

// 事件监听
function initEventListeners() {
    questionInput.addEventListener('input', updateSendButton);
    chatInput.addEventListener('input', updateChatSendButton);
    sendBtn.addEventListener('click', () => sendMessage(questionInput.value));
    chatSendBtn.addEventListener('click', () => sendMessage(chatInput.value));

    questionInput.addEventListener('keydown', (e) => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            if (questionInput.value.trim()) sendMessage(questionInput.value);
        }
    });

    chatInput.addEventListener('keydown', (e) => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            if (chatInput.value.trim()) sendMessage(chatInput.value);
        }
    });

    document.getElementById('add-doc-btn').addEventListener('click', () => {
        document.getElementById('file-input').click();
    });

    document.getElementById('file-input').addEventListener('change', handleFileUpload);
    document.getElementById('select-doc-btn').addEventListener('click', toggleDocList);

    document.addEventListener('click', (e) => {
        if (!e.target.closest('#select-doc-btn') && !e.target.closest('#doc-list')) {
            docList.classList.add('hidden');
        }
        // 点击其他地方关闭处理状态提示
        if (processingStatus === 'completed' && !e.target.closest('#doc-list')) {
            setProcessingStatus('idle');
        }
    });
}

function updateSendButton() { sendBtn.disabled = !questionInput.value.trim(); }
function updateChatSendButton() { chatSendBtn.disabled = !chatInput.value.trim(); }

function setProcessingStatus(status) {
    processingStatus = status;
    processingStatusEl.classList.toggle('hidden', status === 'idle');

    if (status === 'processing') {
        statusTextEl.textContent = '上传文档处理中，请等待';
    } else if (status === 'completed') {
        statusTextEl.textContent = '处理完成，可以选择知识库查询';
    }
}

function quickAsk(question) {
    questionInput.value = question;
    updateSendButton();
    sendMessage(question);
}

async function sendMessage(question) {
    // 防止重复提交
    if (isRequestInProgress) {
        console.log('Request already in progress, skipping');
        return;
    }

    if (!question.trim()) return;

    isRequestInProgress = true;
    const requestStartTime = Date.now();
    console.log(`[${new Date().toISOString()}] Starting request for: "${question.substring(0, 30)}..."`);

    mainView.classList.add('hidden');
    chatView.classList.remove('hidden');
    addMessage('user', question);

    questionInput.value = '';
    chatInput.value = '';
    updateSendButton();
    updateChatSendButton();
    loading.classList.remove('hidden');

    // 添加超时控制（120秒，匹配后端LLM超时）
    const controller = new AbortController();
    const timeoutId = setTimeout(() => {
        console.log('Request timeout, aborting');
        controller.abort();
    }, 120000);

    try {
        // 构建请求体，始终传递 selected_files（空数组表示未选择）
        const requestBody = {
            question,
            session_id: currentSessionId,
            selected_files: selectedFiles  // 空数组表示用户未选择任何知识库
        };

        console.log(`[${new Date().toISOString()}] Sending request...`);

        const fetchStartTime = Date.now();
        const response = await fetch(`${API_BASE}/ask`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(requestBody),
            signal: controller.signal
        });
        const fetchDuration = Date.now() - fetchStartTime;

        clearTimeout(timeoutId);
        console.log(`[${new Date().toISOString()}] Response received in ${fetchDuration}ms, status: ${response.status}`);

        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }

        const data = await response.json();
        const totalDuration = Date.now() - requestStartTime;
        console.log(`[${new Date().toISOString()}] Request completed in ${totalDuration}ms`, {
            success: data.success,
            hasAnswer: !!data.answer,
            sourcesCount: data.sources?.length,
            answerLength: data.answer?.length
        });

        if (data.success && data.answer) {
            currentSessionId = data.session_id || currentSessionId;
            addMessage('assistant', data.answer, data.sources || []);
        } else {
            addMessage('assistant', '抱歉，获取回答时出错了：' + (data.message || '未知错误'));
        }
    } catch (error) {
        clearTimeout(timeoutId);
        const totalDuration = Date.now() - requestStartTime;
        if (error.name === 'AbortError') {
            console.error(`[${new Date().toISOString()}] Request timeout after ${totalDuration}ms`);
            addMessage('assistant', `请求超时（${Math.round(totalDuration/1000)}秒），请稍后重试。`);
        } else {
            console.error(`[${new Date().toISOString()}] Error after ${totalDuration}ms:`, error);
            addMessage('assistant', '网络错误，请检查服务是否运行。错误：' + error.message);
        }
    } finally {
        loading.classList.add('hidden');
        isRequestInProgress = false;
    }
}

function addMessage(role, content, sources = null) {
    const messageDiv = document.createElement('div');
    messageDiv.className = `flex ${role === 'user' ? 'justify-end' : 'justify-start'}`;

    const bubble = document.createElement('div');
    bubble.className = `max-w-[80%] rounded-2xl px-5 py-3 ${
        role === 'user' ? 'chat-bubble-user text-white' : 'bg-gray-100 text-gray-800'
    }`;

    const contentDiv = document.createElement('div');
    contentDiv.className = 'prose prose-sm max-w-none';
    contentDiv.innerHTML = formatContent(content);
    bubble.appendChild(contentDiv);

    if (sources && sources.length > 0) {
        const sourcesDiv = document.createElement('div');
        sourcesDiv.className = 'mt-3 pt-3 border-t border-gray-200/30';
        sourcesDiv.innerHTML = `<p class="text-xs opacity-70 mb-2">参考来源：</p>`;

        sources.forEach((source, index) => {
            const sourceBtn = document.createElement('button');
            sourceBtn.className = 'text-xs opacity-70 hover:opacity-100 underline mr-3';
            sourceBtn.textContent = `[${index + 1}]`;
            sourceBtn.onclick = () => showSourceDetail(source);
            sourcesDiv.appendChild(sourceBtn);
        });

        bubble.appendChild(sourcesDiv);
    }

    messageDiv.appendChild(bubble);
    messageList.appendChild(messageDiv);
    messageList.scrollTop = messageList.scrollHeight;
}

function formatContent(content) {
    return content
        .replace(/\n/g, '<br>')
        .replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>')
        .replace(/\*(.+?)\*/g, '<em>$1</em>')
        .replace(/`(.+?)`/g, '<code class="bg-black/10 px-1 rounded">$1</code>');
}

function showSourceDetail(source) {
    const modal = document.createElement('div');
    modal.className = 'fixed inset-0 bg-black/50 flex items-center justify-center z-50 p-4';

    // 安全的清理逻辑：只移除明确的控制字符和乱码，保留中文
    let cleanContent = (source.content || '');

    // 第一步：移除 null 和控制字符（保留换行\n和回车\r）
    cleanContent = cleanContent
        // 移除 null 字符
        .replace(/\x00/g, '')
        // 移除 C0 控制字符（0-31），但保留\n(10)和\r(13)
        .replace(/[\x01-\x09\x0B\x0C\x0E-\x1F]/g, '')
        // 移除 DEL (127) 和 C1 控制字符 (128-159)
        .replace(/[\x7F-\x9F]/g, '');

    // 第二步：智能分段处理
    // 策略：检测文本是否以双换行分段，如果没有则使用单换行分段
    cleanContent = cleanContent
        // 3个以上换行合并为双换行
        .replace(/\n{3,}/g, '\n\n')
        .trim();

    // 检测分段方式：如果有双换行就用双换行，否则用单换行
    let paragraphs;
    if (cleanContent.includes('\n\n')) {
        // 有双换行，使用双换行分段
        paragraphs = cleanContent.split('\n\n').filter(p => p.trim());
    } else {
        // 没有双换行，使用单换行分段，并将短行合并
        const lines = cleanContent.split('\n').filter(l => l.trim());
        paragraphs = [];
        let currentPara = '';

        for (const line of lines) {
            const trimmedLine = line.trim();
            // 如果行以标点结尾，或者是标题格式（较短且不以标点结尾），作为独立段落
            const isTitle = trimmedLine.length < 30 && !trimmedLine.match(/[。！？；，]$/);
            const endsWithPunct = trimmedLine.match(/[。！？；]$/);

            if (isTitle || endsWithPunct) {
                if (currentPara) {
                    currentPara += ' ' + trimmedLine;
                    paragraphs.push(currentPara.trim());
                    currentPara = '';
                } else {
                    paragraphs.push(trimmedLine);
                }
            } else {
                // 继续累积当前段落
                currentPara = currentPara ? currentPara + ' ' + trimmedLine : trimmedLine;
            }
        }

        // 添加最后一段
        if (currentPara.trim()) {
            paragraphs.push(currentPara.trim());
        }
    }

    // 确保每个段落都是独立显示
    const formattedContent = paragraphs.map(p => {
        // 段落内部多空格合并
        const cleanPara = p.replace(/ {2,}/g, ' ').trim();
        return `<p class="mb-3 last:mb-0 leading-relaxed">${escapeHtml(cleanPara)}</p>`;
    }).join('');

    modal.innerHTML = `
        <div class="bg-white rounded-2xl w-full max-w-2xl max-h-[80vh] flex flex-col shadow-xl" onclick="event.stopPropagation()">
            <div class="p-4 border-b border-gray-200 flex justify-between items-center flex-shrink-0">
                <h3 class="font-semibold text-gray-800">参考来源</h3>
                <button onclick="this.closest('.fixed').remove()" class="text-gray-500 hover:text-gray-700 w-8 h-8 flex items-center justify-center rounded-full hover:bg-gray-100">
                    <i class="fas fa-times"></i>
                </button>
            </div>
            <div class="p-4 overflow-y-auto flex-1">
                <p class="text-sm text-gray-600 mb-3">相关度: <span class="font-medium text-purple-600">${(source.score * 100).toFixed(1)}%</span></p>
                <div class="bg-gray-50 rounded-lg p-4 text-sm text-gray-800 leading-relaxed">${formattedContent || '<p class="text-gray-500">无内容</p>'}</div>
                ${source.metadata?.file_name ? `<p class="text-xs text-gray-500 mt-3 text-right">来源: ${source.metadata.file_name}</p>` : ''}
            </div>
        </div>
    `;
    document.body.appendChild(modal);
    modal.addEventListener('click', (e) => { if (e.target === modal) modal.remove(); });
}

// HTML 转义函数，防止 XSS
function escapeHtml(text) {
    if (!text) return '';
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

function newChat() {
    currentSessionId = null;
    messageList.innerHTML = '';
    chatView.classList.add('hidden');
    mainView.classList.remove('hidden');
    questionInput.focus();
}

async function loadDocuments() {
    const saved = localStorage.getItem('uploadedFiles');
    if (saved) {
        uploadedFiles = JSON.parse(saved);
        updateDocList();
    }
    // 检查服务器索引状态，如果为空则清除本地列表
    try {
        const response = await fetch(`${API_BASE}/health`);
        if (response.ok) {
            const data = await response.json();
            // 如果服务器没有文档，清除前端列表
            if (data.document_count === 0 || !data.has_index) {
                console.log('Server index empty, clearing local file list');
                uploadedFiles = [];
                selectedFiles = [];
                localStorage.removeItem('uploadedFiles');
                updateDocList();
            }
        }
    } catch (e) {
        // 忽略健康检查错误
    }
}

async function handleFileUpload(e) {
    const files = e.target.files;
    if (!files.length) return;

    loading.classList.remove('hidden');

    try {
        // 创建 FormData 上传文件
        const formData = new FormData();
        for (const file of files) {
            formData.append('files', file);
        }

        // 显示处理中状态
        setProcessingStatus('processing');

        // 实际上传文件到后端
        const response = await fetch(`${API_BASE}/upload`, {
            method: 'POST',
            body: formData
        });

        const data = await response.json();

        if (data.success) {
            // 更新本地文件列表
            for (const file of files) {
                if (!uploadedFiles.includes(file.name)) {
                    uploadedFiles.push(file.name);
                }
            }
            localStorage.setItem('uploadedFiles', JSON.stringify(uploadedFiles));
            updateDocList();

            // 处理完成
            setProcessingStatus('completed');
        } else {
            alert('上传失败: ' + data.message);
            setProcessingStatus('idle');
        }

    } catch (error) {
        console.error('Upload error:', error);
        alert('上传失败，请重试');
        setProcessingStatus('idle');
    } finally {
        loading.classList.add('hidden');
        e.target.value = '';
    }
}

async function ingestDocuments() {
    try {
        const response = await fetch(`${API_BASE}/ingest`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ document_path: './knowledge_base', recursive: true })
        });

        const data = await response.json();
        if (data.success) {
            console.log('Documents ingested:', data.message);
        }
    } catch (error) {
        console.error('Ingest error:', error);
    }
}

function toggleDocList() { docList.classList.toggle('hidden'); }

function updateDocList() {
    const content = document.getElementById('doc-list-content');

    if (uploadedFiles.length === 0) {
        content.innerHTML = '<p class="text-gray-400 text-sm px-3 py-2">暂无文档</p>';
        docCount.textContent = '选择知识库';
        return;
    }

    content.innerHTML = uploadedFiles.map(file => `
        <div class="flex items-center justify-between gap-3 px-3 py-2 hover:bg-gray-50 rounded">
            <label class="flex items-center gap-3 cursor-pointer flex-1 min-w-0">
                <input type="checkbox" ${selectedFiles.includes(file) ? 'checked' : ''}
                    onchange="toggleFileSelection('${file}')"
                    class="w-4 h-4 text-purple-500 border-gray-300 rounded focus:ring-purple-500 flex-shrink-0">
                <span class="text-gray-800 text-sm truncate">${file}</span>
            </label>
            <button onclick="showDeleteModal('${file}')"
                class="p-1 hover:bg-gray-200 rounded transition-colors flex-shrink-0">
                <i class="fas fa-trash text-gray-400 text-xs"></i>
            </button>
        </div>
    `).join('');

    docCount.textContent = selectedFiles.length > 0 ? `已选择 ${selectedFiles.length} 个文档` : '选择知识库';
}

function toggleFileSelection(fileName) {
    const index = selectedFiles.indexOf(fileName);
    if (index > -1) selectedFiles.splice(index, 1);
    else selectedFiles.push(fileName);
    updateDocList();
}

// 删除相关功能
function showDeleteModal(fileName) {
    fileToDelete = fileName;
    deleteFileName.textContent = fileName;
    deleteModal.classList.remove('hidden');
}

function closeDeleteModal() {
    fileToDelete = null;
    deleteModal.classList.add('hidden');
}

function confirmDelete() {
    if (!fileToDelete) return;

    // 从列表中移除
    uploadedFiles = uploadedFiles.filter(f => f !== fileToDelete);
    selectedFiles = selectedFiles.filter(f => f !== fileToDelete);

    // 更新本地存储
    localStorage.setItem('uploadedFiles', JSON.stringify(uploadedFiles));

    // 更新UI
    updateDocList();
    closeDeleteModal();
}
