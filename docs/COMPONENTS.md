# HTML 组件化开发文档

## 📁 目录结构

```
docs/
├── index.html                 # 主页面（完整版）
├── index-modular.html         # 模块化版本（使用 include 指令）
├── components/                # HTML 组件目录
│   ├── header.html           # 头部导航组件
│   ├── hero.html             # 首屏展示组件
│   ├── features.html         # 功能特性组件
│   ├── data-showcase.html    # 数据展示组件
│   ├── downloads.html        # 下载中心组件
│   ├── contact.html          # 联系我们组件
│   └── footer.html           # 页脚组件
├── styles/
│   └── main.css              # 主样式表
├── scripts/
│   └── main.js               # 主脚本
└── manifest/
    └── manifest.json         # PWA 清单
```

## 🎯 组件化优势

### 1. **模块化设计**
- 每个组件独立、可复用
- 便于团队协作开发
- 降低代码耦合度

### 2. **语义化 HTML**
- 符合 W3C 标准
- 良好的无障碍访问支持
- 搜索引擎友好

### 3. **BEM 命名规范**
```css
/* Block（块） */
.feature-card { }

/* Element（元素） */
.feature-card__icon { }
.feature-card__title { }

/* Modifier（修饰符） */
.feature-card--highlighted { }
```

### 4. **响应式支持**
- 移动优先设计
- 自适应断点
- 触摸友好

## 🔧 使用方法

### 方式一：SSI Include（推荐用于生产环境）

在 HTML 中使用 SSI 指令：

```html
<!--#include virtual="components/header.html" -->
<!--#include virtual="components/hero.html" -->
<!--#include virtual="components/footer.html" -->
```

**服务器配置：**

#### Apache
```apache
# .htaccess
Options +Includes
AddType text/html .shtml
AddOutputFilter INCLUDES .shtml
```

#### Nginx
```nginx
# nginx.conf
location ~ \.shtml$ {
    ssi on;
    ssi_silent_errors on;
}
```

#### Node.js (Express)
```javascript
app.use(express.static('docs'));
app.use('/components', express.static('docs/components'));
```

### 方式二：JavaScript Include（开发环境）

```html
<div id="header"></div>
<script>
    fetch('components/header.html')
        .then(response => response.text())
        .then(html => {
            document.getElementById('header').innerHTML = html;
        });
</script>
```

### 方式三：构建工具（推荐用于大型项目）

使用 Webpack、Gulp 或 Grunt 等工具在构建时合并组件。

**Webpack 示例：**
```javascript
// webpack.config.js
module.exports = {
  module: {
    rules: [
      {
        test: /\.html$/,
        use: ['html-loader']
      }
    ]
  }
};
```

## 📦 组件说明

### 1. Header 组件 (`components/header.html`)

**功能：**
- 品牌 Logo 展示
- 主导航菜单
- 移动端汉堡菜单
- 面包屑导航

**无障碍特性：**
- ARIA 角色标注
- 键盘导航支持
- 焦点管理

**CSS 钩子：**
```css
.header { }
.header__container { }
.header__brand { }
.header__logo { }
.header__title { }
.header__toggle { }
.nav { }
.nav__list { }
.nav__item { }
.nav__link { }
```

### 2. Hero 组件 (`components/hero.html`)

**功能：**
- 主标题和副标题
- 描述文本
- CTA 按钮组
- 关键统计数据
- 动画图表展示

**响应式断点：**
```css
/* 移动端 */
@media (max-width: 768px) {
    .hero__container {
        flex-direction: column;
    }
}

/* 平板 */
@media (min-width: 769px) and (max-width: 1024px) {
    .hero__content {
        width: 60%;
    }
}

/* 桌面端 */
@media (min-width: 1025px) {
    .hero__container {
        max-width: 1200px;
    }
}
```

### 3. Features 组件 (`components/features.html`)

**功能：**
- 功能卡片网格
- 图标展示
- 悬停效果
- 键盘可访问

**网格布局：**
```css
.features__grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
    gap: 2rem;
}
```

### 4. Data Showcase 组件 (`components/data-showcase.html`)

**功能：**
- Tab 切换导航
- 多面板数据展示
- 统计数据卡片
- 操作按钮组

**JavaScript 交互：**
```javascript
// Tab 切换逻辑
const tabs = document.querySelectorAll('.data-tab');
const panels = document.querySelectorAll('.data-panel');

tabs.forEach(tab => {
    tab.addEventListener('click', () => {
        // 移除所有 active 状态
        tabs.forEach(t => {
            t.setAttribute('aria-selected', 'false');
            t.classList.remove('data-tab--active');
        });
        panels.forEach(p => {
            p.hidden = true;
            p.classList.remove('data-panel--active');
        });
        
        // 激活当前 tab
        tab.setAttribute('aria-selected', 'true');
        tab.classList.add('data-tab--active');
        const panelId = tab.getAttribute('aria-controls');
        const panel = document.getElementById(panelId);
        panel.hidden = false;
        panel.classList.add('data-panel--active');
    });
});
```

### 5. Downloads 组件 (`components/downloads.html`)

**功能：**
- 文件列表展示
- 分类筛选
- 格式标识
- 下载次数统计
- 懒加载支持

**筛选功能：**
```javascript
// 文件筛选
const filters = document.querySelectorAll('.downloads__filter');
const items = document.querySelectorAll('.download-item');

filters.forEach(filter => {
    filter.addEventListener('click', () => {
        const category = filter.dataset.filter;
        
        items.forEach(item => {
            if (category === 'all' || item.dataset.category === category) {
                item.style.display = 'block';
            } else {
                item.style.display = 'none';
            }
        });
    });
});
```

### 6. Contact 组件 (`components/contact.html`)

**功能：**
- 联系信息展示
- 在线表单
- 表单验证
- 实时反馈

**表单验证：**
```javascript
// 表单验证逻辑
const form = document.getElementById('contact-form');

form.addEventListener('submit', (e) => {
    e.preventDefault();
    
    const name = form.querySelector('[name="name"]').value.trim();
    const email = form.querySelector('[name="email"]').value.trim();
    const message = form.querySelector('[name="message"]').value.trim();
    
    // 验证
    if (!name) {
        showError('name', '请输入您的姓名');
        return;
    }
    
    if (!isValidEmail(email)) {
        showError('email', '请输入有效的邮箱地址');
        return;
    }
    
    if (!message) {
        showError('message', '请输入留言内容');
        return;
    }
    
    // 提交
    submitForm();
});

function isValidEmail(email) {
    return /^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(email);
}
```

### 7. Footer 组件 (`components/footer.html`)

**功能：**
- 版权信息
- 快速链接
- 联系方式
- 社交链接
- 备案信息
- 返回顶部按钮

## 🎨 样式定制

### CSS 变量（主题定制）

```css
:root {
    /* 主色调 */
    --color-primary: #2563eb;
    --color-primary-dark: #1d4ed8;
    --color-primary-light: #3b82f6;
    
    /* 背景色 */
    --color-bg: #ffffff;
    --color-bg-secondary: #f8fafc;
    
    /* 文字颜色 */
    --color-text: #1e293b;
    --color-text-secondary: #64748b;
    
    /* 边框 */
    --color-border: #e2e8f0;
    
    /* 间距 */
    --spacing-xs: 0.5rem;
    --spacing-sm: 1rem;
    --spacing-md: 1.5rem;
    --spacing-lg: 2rem;
    --spacing-xl: 3rem;
    
    /* 圆角 */
    --radius-sm: 0.375rem;
    --radius-md: 0.5rem;
    --radius-lg: 0.75rem;
    
    /* 阴影 */
    --shadow-sm: 0 1px 2px 0 rgba(0, 0, 0, 0.05);
    --shadow-md: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
    --shadow-lg: 0 10px 15px -3px rgba(0, 0, 0, 0.1);
}
```

### 暗色主题支持

```css
@media (prefers-color-scheme: dark) {
    :root {
        --color-bg: #0f172a;
        --color-bg-secondary: #1e293b;
        --color-text: #f1f5f9;
        --color-text-secondary: #94a3b8;
        --color-border: #334155;
    }
}
```

## ⚡ 性能优化

### 1. 资源加载优先级

```html
<!-- 关键 CSS 内联 -->
<style>
    /* 首屏关键样式 */
</style>

<!-- 非关键 CSS 异步加载 -->
<link rel="preload" href="styles/main.css" as="style" onload="this.onload=null;this.rel='stylesheet'">
<noscript><link rel="stylesheet" href="styles/main.css"></noscript>

<!-- JavaScript 延迟加载 -->
<script src="scripts/main.js" type="module" defer></script>
```

### 2. 图片懒加载

```html
<img src="placeholder.jpg" 
     data-src="actual-image.jpg" 
     loading="lazy" 
     alt="描述">
```

### 3. 字体优化

```css
@font-face {
    font-family: 'Noto Sans SC';
    font-display: swap; /* 避免字体加载阻塞 */
}
```

## ♿ 无障碍访问

### ARIA 角色使用

```html
<!-- 导航 -->
<nav role="navigation" aria-label="主导航">

<!-- 标签页 -->
<div role="tablist" aria-label="数据类型选择">
    <button role="tab" aria-selected="true" aria-controls="panel1">
    <div role="tabpanel" id="panel1">

<!-- 表单 -->
<label for="email">邮箱</label>
<input type="email" id="email" aria-required="true" aria-describedby="email-help">
<small id="email-help">我们不会泄露您的邮箱地址</small>
```

### 键盘导航

```javascript
// Tab 键导航
element.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' || e.key === ' ') {
        e.preventDefault();
        element.click();
    }
    if (e.key === 'Escape') {
        closeMenu();
    }
});
```

## 📊 SEO 优化

### 结构化数据

```html
<script type="application/ld+json">
{
    "@context": "https://schema.org",
    "@type": "GovernmentOrganization",
    "name": "National Stats",
    "description": "宏观经济数据分析平台",
    "url": "https://badhope.github.io/national_stats"
}
</script>
```

### Meta 标签

```html
<meta name="description" content="专业的宏观经济数据分析平台">
<meta name="keywords" content="宏观经济，数据分析，GDP,CPI">
<meta name="author" content="National Stats Team">
<meta name="robots" content="index, follow">

<!-- Open Graph -->
<meta property="og:title" content="宏观经济数据分析平台">
<meta property="og:description" content="专业的数据分析与可视化工具">
<meta property="og:image" content="og-image.jpg">
<meta property="og:url" content="https://badhope.github.io/national_stats">

<!-- Twitter Card -->
<meta name="twitter:card" content="summary_large_image">
```

## 🧪 测试

### HTML 验证

```bash
# 使用 W3C 验证器
https://validator.w3.org/

# 本地验证（Node.js）
npm install -g htmlhint
htmlhint docs/*.html
```

### 无障碍测试

```bash
# 使用 axe-core
npm install -g @axe-core/cli
axe http://localhost:8080
```

### 性能测试

```bash
# 使用 Lighthouse CI
npm install -g @lhci/cli
lhci autorun
```

## 📝 更新日志

### v2.0.0 (2024-01-20)
- ✅ 完成 HTML 组件化重构
- ✅ 添加完整的 BEM 命名
- ✅ 增强无障碍支持
- ✅ 优化响应式布局
- ✅ 添加组件使用文档

### v1.0.0 (2024-01-15)
- 初始版本发布

## 🤝 贡献指南

### 添加新组件

1. 在 `components/` 目录创建新文件
2. 遵循 BEM 命名规范
3. 添加完整的 ARIA 标注
4. 编写组件文档
5. 测试响应式和键盘导航

### 代码审查清单

- [ ] HTML 语义化正确
- [ ] ARIA 角色完整
- [ ] 键盘导航可用
- [ ] 响应式布局正常
- [ ] 浏览器兼容性良好
- [ ] 性能优化到位
- [ ] 文档齐全

## 📧 联系方式

如有问题或建议，请联系：
- 📧 Email: support@nationalstats.gov.cn
- 📱 Phone: 010-12345678

---

**最后更新：** 2024-01-20  
**维护团队：** National Stats Team
