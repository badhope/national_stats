# 宏观经济数据分析平台 - 部署文档

## 一、系统要求

### 1.1 运行环境
- **操作系统**: Windows Server 2019+ / Linux (Ubuntu 20.04+)
- **Web服务器**: Nginx 1.18+ / Apache 2.4+
- **Node.js**: 18.0+ (如需构建)
- **浏览器支持**: Chrome 90+, Firefox 88+, Safari 14+, Edge 90+

### 1.2 网络要求
- 带宽: 最低 5Mbps
- 域名: 已备案域名（如需国内访问）

---

## 二、文件结构

```
docs/
├── index.html          # 主页面
├── styles/
│   └── main.css        # 样式文件
└── scripts/
    └── main.js         # 交互脚本
```

---

## 三、部署步骤

### 3.1 Nginx 配置

```nginx
server {
    listen 80;
    server_name your-domain.com;
    root /var/www/docs;
    index index.html;

    # Gzip 压缩
    gzip on;
    gzip_types text/plain text/css application/json application/javascript text/xml application/xml;

    # 缓存策略
    location ~* \.(js|css|png|jpg|jpeg|gif|ico|svg)$ {
        expires 30d;
        add_header Cache-Control "public, immutable";
    }

    # 安全头
    add_header X-Frame-Options "SAMEORIGIN" always;
    add_header X-Content-Type-Options "nosniff" always;
    add_header X-XSS-Protection "1; mode=block" always;

    # SPA 路由支持
    location / {
        try_files $uri $uri/ /index.html;
    }
}
```

### 3.2 Apache 配置 (.htaccess)

```apache
<IfModule mod_rewrite.c>
    RewriteEngine On
    RewriteBase /
    RewriteRule ^index\.html$ - [L]
    RewriteCond %{REQUEST_FILENAME} !-f
    RewriteCond %{REQUEST_FILENAME} !-d
    RewriteRule . /index.html [L]
</IfModule>

# 缓存配置
<IfModule mod_expires.c>
    ExpiresActive On
    ExpiresByType text/css "access plus 30 days"
    ExpiresByType application/javascript "access plus 30 days"
    ExpiresByType image/png "access plus 30 days"
</IfModule>
```

---

## 四、功能说明

### 4.1 核心功能
- **响应式设计**: 适配移动端、平板、桌面设备
- **下载功能**: 支持直接下载和新窗口打开两种模式
- **导航系统**: 固定头部导航 + 面包屑导航 + 返回顶部按钮
- **下载反馈**: 加载动画、成功/失败状态提示

### 4.2 交互特性
- 移动端汉堡菜单
- 下拉子菜单
- 平滑滚动
- 键盘导航支持

---

## 五、浏览器兼容性

| 浏览器 | 最新版本 | 前一版本 | 前两版本 |
|--------|----------|----------|----------|
| Chrome | ✅ | ✅ | ✅ |
| Firefox | ✅ | ✅ | ✅ |
| Safari | ✅ | ✅ | ✅ |
| Edge | ✅ | ✅ | ✅ |

---

## 六、性能指标

- **FCP (首次内容绘制)**: < 1.8秒
- **LCP (最大内容绘制)**: < 2.5秒
- **TBT (总阻塞时间)**: < 200毫秒
- **CLS (累积布局偏移)**: < 0.1

---

## 七、安全配置

### 7.1 HTTPS 配置
```nginx
# 使用 Let's Encrypt 免费证书
sudo apt install certbot python3-certbot-nginx
sudo certbot --nginx -d your-domain.com
```

### 7.2 安全头
```nginx
add_header Content-Security-Policy "default-src 'self';" always;
add_header Referrer-Policy "strict-origin-when-cross-origin" always;
```

---

## 八、测试检查清单

- [ ] 所有导航链接正常工作
- [ ] 响应式布局在各断点正常显示
- [ ] 下载按钮功能正常
- [ ] 返回顶部按钮显示/隐藏正常
- [ ] 面包屑导航正确显示
- [ ] 移动端菜单正常切换
- [ ] 键盘导航正常工作
- [ ] 无控制台错误

---

## 九、常见问题

### Q1: 下载功能无法使用
A: 确保服务器正确配置了静态文件 MIME 类型

### Q2: 移动端菜单无法打开
A: 检查 JavaScript 是否正确加载，确认无 CSP 阻止

### Q3: 字体加载失败
A: 检查 Google Fonts 是否可访问，或使用本地字体

---

## 十、联系方式

如有问题，请联系技术支持团队。
