class Navigation {
    constructor() {
        this.header = document.querySelector('.header');
        this.navToggle = document.querySelector('.header__toggle');
        this.nav = document.querySelector('.nav');
        this.navLinks = document.querySelectorAll('.nav__link');
        this.backToTop = document.querySelector('.back-to-top');
        this.submenuItems = document.querySelectorAll('.nav__item--has-submenu');
        
        this.init();
    }
    
    init() {
        this.setupMobileMenu();
        this.setupBackToTop();
        this.setupScrollSpy();
        this.setupSubmenu();
    }
    
    setupMobileMenu() {
        if (!this.navToggle || !this.nav) return;
        
        this.navToggle.addEventListener('click', (e) => {
            e.stopPropagation();
            const isOpen = this.nav.classList.toggle('nav--open');
            this.navToggle.setAttribute('aria-expanded', isOpen);
            document.body.classList.toggle('body--nav-open', isOpen);
            
            if (isOpen) {
                const firstLink = this.nav.querySelector('.nav__link');
                firstLink?.focus();
            }
        });
        
        document.addEventListener('click', (e) => {
            if (!this.nav.contains(e.target) && !this.navToggle.contains(e.target)) {
                this.closeMobileMenu();
            }
        });
        
        this.navLinks.forEach(link => {
            link.addEventListener('keydown', (e) => {
                if (e.key === 'Escape') {
                    this.closeMobileMenu();
                    this.navToggle.focus();
                }
            });
        });
        
        document.addEventListener('keydown', (e) => {
            if (e.key === 'Escape' && this.nav.classList.contains('nav--open')) {
                this.closeMobileMenu();
            }
        });
    }
    
    closeMobileMenu() {
        this.nav.classList.remove('nav--open');
        this.navToggle.setAttribute('aria-expanded', 'false');
        document.body.classList.remove('body--nav-open');
    }
    
    setupSubmenu() {
        this.submenuItems.forEach(item => {
            const link = item.querySelector('.nav__link');
            const submenu = item.querySelector('.nav__submenu');
            
            if (link && submenu) {
                link.addEventListener('click', (e) => {
                    if (window.innerWidth <= 768) {
                        e.preventDefault();
                        const isExpanded = link.getAttribute('aria-expanded') === 'true';
                        link.setAttribute('aria-expanded', !isExpanded);
                        item.classList.toggle('nav__item--submenu-open', !isExpanded);
                    }
                });
                
                link.addEventListener('keydown', (e) => {
                    if (e.key === 'ArrowDown' && submenu) {
                        e.preventDefault();
                        const firstSubmenuLink = submenu.querySelector('.nav__submenu-link');
                        firstSubmenuLink?.focus();
                    }
                });
                
                const submenuLinks = submenu.querySelectorAll('.nav__submenu-link');
                submenuLinks.forEach((sublink, index) => {
                    sublink.addEventListener('keydown', (e) => {
                        if (e.key === 'ArrowDown') {
                            e.preventDefault();
                            const nextLink = submenuLinks[index + 1];
                            nextLink?.focus();
                        }
                        if (e.key === 'ArrowUp') {
                            e.preventDefault();
                            const prevLink = submenuLinks[index - 1] || link;
                            prevLink.focus();
                        }
                        if (e.key === 'Escape') {
                            link.focus();
                            link.setAttribute('aria-expanded', 'false');
                            item.classList.remove('nav__item--submenu-open');
                        }
                    });
                });
            }
        });
    }
    
    setupBackToTop() {
        if (!this.backToTop) return;
        
        const scrollThreshold = 500;
        
        const toggleVisibility = () => {
            if (window.scrollY > scrollThreshold) {
                this.backToTop.classList.add('back-to-top--visible');
            } else {
                this.backToTop.classList.remove('back-to-top--visible');
            }
        };
        
        let ticking = false;
        window.addEventListener('scroll', () => {
            if (!ticking) {
                requestAnimationFrame(() => {
                    toggleVisibility();
                    ticking = false;
                });
                ticking = true;
            }
        }, { passive: true });
        
        this.backToTop.addEventListener('click', () => {
            window.scrollTo({
                top: 0,
                behavior: 'smooth'
            });
            this.backToTop.setAttribute('aria-hidden', 'true');
        });
        
        this.backToTop.addEventListener('keydown', (e) => {
            if (e.key === 'Enter' || e.key === ' ') {
                e.preventDefault();
                this.backToTop.click();
            }
        });
        
        window.addEventListener('scroll', () => {
            if (window.scrollY < 100) {
                this.backToTop.setAttribute('aria-hidden', 'true');
            } else {
                this.backToTop.setAttribute('aria-hidden', 'false');
            }
        }, { passive: true });
    }
    
    setupScrollSpy() {
        const sections = document.querySelectorAll('section[id]');
        
        const observer = new IntersectionObserver((entries) => {
            entries.forEach(entry => {
                if (entry.isIntersecting) {
                    const id = entry.target.getAttribute('id');
                    this.updateActiveNavLink(id);
                }
            });
        }, {
            rootMargin: '-20% 0px -60% 0px',
            threshold: 0
        });
        
        sections.forEach(section => observer.observe(section));
    }
    
    updateActiveNavLink(activeId) {
        this.navLinks.forEach(link => {
            link.classList.remove('nav__link--active');
            const href = link.getAttribute('href');
            if (href === `#${activeId}`) {
                link.classList.add('nav__link--active');
            }
        });
    }
}

class DataTabs {
    constructor() {
        this.tabs = document.querySelectorAll('.data-tab');
        this.panels = document.querySelectorAll('.data-panel');
        
        if (this.tabs.length > 0) {
            this.init();
        }
    }
    
    init() {
        this.tabs.forEach((tab, index) => {
            tab.addEventListener('click', () => this.switchTab(tab));
            tab.addEventListener('keydown', (e) => {
                if (e.key === 'ArrowRight') {
                    e.preventDefault();
                    const nextTab = this.tabs[(index + 1) % this.tabs.length];
                    nextTab?.focus();
                    this.switchTab(nextTab);
                }
                if (e.key === 'ArrowLeft') {
                    e.preventDefault();
                    const prevTab = this.tabs[(index - 1 + this.tabs.length) % this.tabs.length];
                    prevTab?.focus();
                    this.switchTab(prevTab);
                }
            });
        });
        
        this.tabs[0]?.setAttribute('tabindex', '0');
    }
    
    switchTab(selectedTab) {
        const targetId = selectedTab.getAttribute('aria-controls');
        
        this.tabs.forEach(tab => {
            tab.classList.remove('active');
            tab.setAttribute('aria-selected', 'false');
            tab.setAttribute('tabindex', '-1');
        });
        
        this.panels.forEach(panel => {
            panel.classList.remove('active');
        });
        
        selectedTab.classList.add('active');
        selectedTab.setAttribute('aria-selected', 'true');
        selectedTab.setAttribute('tabindex', '0');
        
        const targetPanel = document.getElementById(targetId);
        if (targetPanel) {
            targetPanel.classList.add('active');
            targetPanel.focus();
        }
        
        this.announceChange(selectedTab.textContent.trim());
    }
    
    announceChange(tabName) {
        let announcer = document.getElementById('tab-announcer');
        if (!announcer) {
            announcer = document.createElement('div');
            announcer.id = 'tab-announcer';
            announcer.setAttribute('aria-live', 'polite');
            announcer.setAttribute('aria-atomic', 'true');
            announcer.className = 'sr-only';
            document.body.appendChild(announcer);
        }
        announcer.textContent = `已切换到${tabName}标签`;
    }
}

class DownloadManager {
    constructor() {
        this.downloadButtons = document.querySelectorAll('.download-btn');
        this.toastContainer = document.querySelector('.toast-container');
        this.downloads = new Map();
        
        this.init();
    }
    
    init() {
        this.downloadButtons.forEach(btn => {
            btn.addEventListener('click', (e) => this.handleDownload(e));
            btn.addEventListener('keydown', (e) => {
                if (e.key === 'Enter' || e.key === ' ') {
                    e.preventDefault();
                    btn.click();
                }
            });
        });
    }
    
    async handleDownload(event) {
        const button = event.currentTarget;
        const filePath = button.dataset.file;
        const fileName = button.dataset.filename;
        const mode = button.dataset.mode || 'direct';
        
        if (button.classList.contains('download-btn--loading') ||
            button.classList.contains('download-btn--success')) {
            return;
        }
        
        this.setButtonLoading(button, true);
        
        try {
            const exists = await this.checkFileExists(filePath);
            
            if (!exists) {
                throw new Error('文件不存在或已被移除');
            }
            
            await this.simulateDownload(button, fileName);
            
            this.setButtonSuccess(button);
            this.showToast(`已成功下载: ${fileName}`, 'success');
            
            if (mode === 'new-window') {
                window.open(filePath, '_blank');
            }
            
        } catch (error) {
            this.setButtonError(button);
            this.showToast(error.message || '下载失败，请重试', 'error');
            
            setTimeout(() => {
                this.resetButton(button);
            }, 3000);
        }
    }
    
    async checkFileExists(filePath) {
        return new Promise((resolve) => {
            setTimeout(() => {
                const mockFiles = [
                    'reports/annual_2024.pdf',
                    'data/gdp_yoy_1990_2024.csv',
                    'data/cpi_index.xlsx',
                    'reports/employment_2024.pdf'
                ];
                resolve(mockFiles.includes(filePath));
            }, 500);
        });
    }
    
    simulateDownload(button, fileName) {
        return new Promise((resolve) => {
            const duration = 1500 + Math.random() * 1500;
            const progress = button.querySelector('.download-btn__progress circle');
            
            let startTime = null;
            
            const animate = (timestamp) => {
                if (!startTime) startTime = timestamp;
                const elapsed = timestamp - startTime;
                const progressValue = Math.min((elapsed / duration) * 100, 100);
                
                if (progress) {
                    const circumference = 2 * Math.PI * 10;
                    const offset = circumference - (progressValue / 100) * circumference;
                    progress.style.strokeDashoffset = offset;
                }
                
                if (elapsed < duration) {
                    requestAnimationFrame(animate);
                } else {
                    resolve();
                }
            };
            
            requestAnimationFrame(animate);
        });
    }
    
    setButtonLoading(button, loading) {
        if (loading) {
            button.classList.add('download-btn--loading');
            button.classList.remove('download-btn--success', 'download-btn--error');
            button.setAttribute('aria-busy', 'true');
        } else {
            button.classList.remove('download-btn--loading');
            button.setAttribute('aria-busy', 'false');
        }
    }
    
    setButtonSuccess(button) {
        button.classList.remove('download-btn--loading', 'download-btn--error');
        button.classList.add('download-btn--success');
        
        const icon = button.querySelector('.download-btn__icon');
        const text = button.querySelector('.download-btn__text');
        
        if (icon) {
            icon.innerHTML = '<path d="M9 16.17L4.83 12l-1.42 1.41L9 19 21 7l-1.41-1.41z" fill="currentColor"/>';
        }
        if (text) {
            text.textContent = '下载完成';
        }
        
        button.setAttribute('aria-busy', 'false');
        
        setTimeout(() => {
            this.resetButton(button);
        }, 3000);
    }
    
    setButtonError(button) {
        button.classList.remove('download-btn--loading');
        button.classList.add('download-btn--error');
        button.setAttribute('aria-busy', 'false');
        
        const icon = button.querySelector('.download-btn__icon');
        const text = button.querySelector('.download-btn__text');
        
        if (icon) {
            icon.innerHTML = '<path d="M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2zm1 15h-2v-2h2v2zm0-4h-2V7h2v6z" fill="currentColor"/>';
        }
        if (text) {
            text.textContent = '下载失败';
        }
    }
    
    resetButton(button) {
        button.classList.remove('download-btn--loading', 'download-btn--success', 'download-btn--error');
        button.removeAttribute('aria-busy');
        
        const icon = button.querySelector('.download-btn__icon');
        const text = button.querySelector('.download-btn__text');
        const progress = button.querySelector('.download-btn__progress circle');
        
        if (icon) {
            icon.innerHTML = '<path d="M19 9h-4V3H9v6H5l7 7 7-7zM5 18v2h14v-2H5z" fill="currentColor"/>';
        }
        if (text) {
            const originalText = button.dataset.mode === 'new-window' ? '新窗口打开' : '立即下载';
            text.textContent = originalText;
        }
        if (progress) {
            progress.style.strokeDashoffset = '31.4';
        }
    }
    
    showToast(message, type = 'info') {
        if (!this.toastContainer) return;
        
        const toast = document.createElement('div');
        toast.className = `toast toast--${type}`;
        toast.setAttribute('role', 'alert');
        
        const icons = {
            success: '<path d="M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2zm-2 15l-5-5 1.41-1.41L10 14.17l7.59-7.59L19 8l-9 9z" fill="currentColor"/>',
            error: '<path d="M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2zm1 15h-2v-2h2v2zm0-4h-2V7h2v6z" fill="currentColor"/>',
            warning: '<path d="M1 21h22L12 2 1 21zm12-3h-2v-2h2v2zm0-4h-2v-4h2v4z" fill="currentColor"/>',
            info: '<path d="M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2zm1 15h-2v-6h2v6zm0-8h-2V7h2v2z" fill="currentColor"/>'
        };
        
        toast.innerHTML = `
            <svg class="toast__icon" viewBox="0 0 24 24" aria-hidden="true">
                ${icons[type] || icons.info}
            </svg>
            <span class="toast__message">${message}</span>
            <button class="toast__close" aria-label="关闭提示">
                <svg viewBox="0 0 24 24"><path d="M19 6.41L17.59 5 12 10.59 6.41 5 5 6.41 10.59 12 5 17.59 6.41 19 12 13.41 17.59 19 19 17.59 13.41 12z" fill="currentColor"/></svg>
            </button>
        `;
        
        const closeBtn = toast.querySelector('.toast__close');
        closeBtn?.addEventListener('click', () => {
            toast.classList.add('toast--hiding');
            toast.addEventListener('animationend', () => toast.remove());
        });
        
        this.toastContainer.appendChild(toast);
        
        requestAnimationFrame(() => {
            toast.classList.add('toast--visible');
        });
        
        setTimeout(() => {
            if (toast.parentNode) {
                toast.classList.add('toast--hiding');
                toast.addEventListener('animationend', () => {
                    toast.remove();
                });
            }
        }, 4000);
    }
}

class ContactForm {
    constructor() {
        this.form = document.getElementById('contact-form');
        this.fields = {
            name: {
                element: document.getElementById('name'),
                errorElement: document.getElementById('name-error'),
                validate: (value) => {
                    if (!value.trim()) return '请输入您的姓名';
                    if (value.trim().length < 2) return '姓名至少需要2个字符';
                    return null;
                }
            },
            email: {
                element: document.getElementById('email'),
                errorElement: document.getElementById('email-error'),
                validate: (value) => {
                    if (!value.trim()) return '请输入您的邮箱';
                    const emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
                    if (!emailRegex.test(value)) return '请输入有效的邮箱地址';
                    return null;
                }
            },
            subject: {
                element: document.getElementById('subject'),
                errorElement: document.getElementById('subject-error'),
                validate: (value) => {
                    if (!value) return '请选择咨询主题';
                    return null;
                }
            },
            message: {
                element: document.getElementById('message'),
                errorElement: document.getElementById('message-error'),
                validate: (value) => {
                    if (!value.trim()) return '请输入留言内容';
                    if (value.trim().length < 10) return '留言内容至少需要10个字符';
                    return null;
                }
            },
            privacy: {
                element: document.getElementById('privacy'),
                errorElement: null,
                validate: (value, element) => {
                    if (!element.checked) return '请阅读并同意隐私政策';
                    return null;
                }
            }
        };
        
        if (this.form) {
            this.init();
        }
    }
    
    init() {
        this.form.setAttribute('novalidate', 'true');
        
        this.form.addEventListener('submit', (e) => this.handleSubmit(e));
        
        Object.values(this.fields).forEach(field => {
            if (field.element) {
                field.element.addEventListener('blur', () => this.validateField(field));
                field.element.addEventListener('input', () => {
                    if (field.element.classList.contains('contact-form__input--error')) {
                        this.validateField(field);
                    }
                });
            }
        });
        
        const submitBtn = this.form.querySelector('.contact-form__submit');
        if (submitBtn) {
            submitBtn.addEventListener('click', () => {
                this.form.reportValidity();
            });
        }
    }
    
    validateField(field) {
        const value = field.element?.value || '';
        const error = field.validate(value, field.element);
        
        if (error) {
            this.showError(field, error);
            return false;
        } else {
            this.clearError(field);
            return true;
        }
    }
    
    showError(field, message) {
        if (field.element) {
            field.element.classList.add('contact-form__input--error');
            field.element.setAttribute('aria-invalid', 'true');
        }
        if (field.errorElement) {
            field.errorElement.textContent = message;
            field.errorElement.setAttribute('role', 'alert');
        }
    }
    
    clearError(field) {
        if (field.element) {
            field.element.classList.remove('contact-form__input--error');
            field.element.setAttribute('aria-invalid', 'false');
        }
        if (field.errorElement) {
            field.errorElement.textContent = '';
        }
    }
    
    async handleSubmit(e) {
        e.preventDefault();
        
        let isValid = true;
        const formData = {};
        
        for (const [name, field] of Object.entries(this.fields)) {
            if (!this.validateField(field)) {
                isValid = false;
            } else {
                formData[name] = field.element?.value;
            }
        }
        
        if (!isValid) {
            const firstError = this.form.querySelector('.contact-form__input--error, .contact-form__select--error');
            firstError?.focus();
            return;
        }
        
        const submitBtn = this.form.querySelector('.contact-form__submit');
        const originalText = submitBtn?.innerHTML;
        
        if (submitBtn) {
            submitBtn.disabled = true;
            submitBtn.innerHTML = '<span>提交中...</span>';
            submitBtn.setAttribute('aria-busy', 'true');
        }
        
        await this.simulateSubmit(formData);
        
        if (submitBtn) {
            submitBtn.disabled = false;
            submitBtn.innerHTML = originalText;
            submitBtn.setAttribute('aria-busy', 'false');
        }
        
        this.form.reset();
        this.showSuccessMessage();
    }
    
    simulateSubmit(data) {
        return new Promise((resolve) => {
            console.log('Form submitted:', data);
            setTimeout(resolve, 1500);
        });
    }
    
    showSuccessMessage() {
        const toastContainer = document.querySelector('.toast-container');
        if (toastContainer) {
            const toast = document.createElement('div');
            toast.className = 'toast toast--success';
            toast.setAttribute('role', 'alert');
            toast.innerHTML = `
                <svg class="toast__icon" viewBox="0 0 24 24">
                    <path d="M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2zm-2 15l-5-5 1.41-1.41L10 14.17l7.59-7.59L19 8l-9 9z" fill="currentColor"/>
                </svg>
                <span class="toast__message">提交成功！我们会尽快与您联系。</span>
            `;
            toastContainer.appendChild(toast);
            
            setTimeout(() => {
                toast.classList.add('toast--hiding');
                toast.addEventListener('animationend', () => toast.remove());
            }, 4000);
        }
    }
}

class Modal {
    constructor() {
        this.modalTriggers = document.querySelectorAll('[data-modal-trigger]');
        this.modals = document.querySelectorAll('.modal');
        this.activeModal = null;
        
        if (this.modals.length > 0) {
            this.init();
        }
    }
    
    init() {
        this.modalTriggers.forEach(trigger => {
            trigger.addEventListener('click', (e) => {
                e.preventDefault();
                const modalId = trigger.dataset.modalTrigger;
                this.openModal(modalId);
            });
        });
        
        this.modals.forEach(modal => {
            const closeButtons = modal.querySelectorAll('[data-modal-close]');
            closeButtons.forEach(btn => {
                btn.addEventListener('click', () => this.closeModal(modal));
            });
            
            modal.addEventListener('click', (e) => {
                if (e.target === modal) {
                    this.closeModal(modal);
                }
            });
            
            modal.addEventListener('keydown', (e) => {
                if (e.key === 'Escape') {
                    this.closeModal(modal);
                }
            });
        });
    }
    
    openModal(modalId) {
        const modal = document.getElementById(modalId);
        if (!modal) return;
        
        this.activeModal = modal;
        modal.classList.add('modal--open');
        modal.setAttribute('aria-hidden', 'false');
        document.body.classList.add('body--modal-open');
        
        const focusableElements = modal.querySelectorAll(
            'button, [href], input, select, textarea, [tabindex]:not([tabindex="-1"])'
        );
        const firstFocusable = focusableElements[0];
        const lastFocusable = focusableElements[focusableElements.length - 1];
        
        firstFocusable?.focus();
        
        modal._handleKeyDown = (e) => {
            if (e.key === 'Tab') {
                if (e.shiftKey && document.activeElement === firstFocusable) {
                    e.preventDefault();
                    lastFocusable?.focus();
                } else if (!e.shiftKey && document.activeElement === lastFocusable) {
                    e.preventDefault();
                    firstFocusable?.focus();
                }
            }
        };
        
        modal.addEventListener('keydown', modal._handleKeyDown);
    }
    
    closeModal(modal) {
        if (!modal) return;
        
        modal.classList.remove('modal--open');
        modal.setAttribute('aria-hidden', 'true');
        document.body.classList.remove('body--modal-open');
        
        if (modal._handleKeyDown) {
            modal.removeEventListener('keydown', modal._handleKeyDown);
        }
        
        this.activeModal = null;
        
        const trigger = document.querySelector(`[data-modal-trigger="${modal.id}"]`);
        trigger?.focus();
    }
}

class SmoothScroll {
    constructor() {
        this.init();
    }
    
    init() {
        document.querySelectorAll('a[href^="#"]').forEach(anchor => {
            anchor.addEventListener('click', (e) => this.handleClick(e));
        });
    }
    
    handleClick(e) {
        const href = e.currentTarget.getAttribute('href');
        if (href === '#') return;
        
        const target = document.querySelector(href);
        if (target) {
            e.preventDefault();
            
            const header = document.querySelector('.header');
            const headerHeight = header?.offsetHeight || 0;
            const targetPosition = target.getBoundingClientRect().top + window.scrollY - headerHeight;
            
            window.scrollTo({
                top: targetPosition,
                behavior: 'smooth'
            });
            
            target.setAttribute('tabindex', '-1');
            target.focus({ preventScroll: true });
            
            history.pushState(null, '', href);
        }
    }
}

class Accessibility {
    constructor() {
        this.init();
    }
    
    init() {
        this.announcePageLoad();
        this.setupFocusVisible();
        this.setupSkipLink();
    }
    
    announcePageLoad() {
        const main = document.querySelector('main');
        if (main && !main.getAttribute('aria-live')) {
            main.setAttribute('aria-live', 'polite');
        }
    }
    
    setupFocusVisible() {
        document.addEventListener('keydown', (e) => {
            if (e.key === 'Tab') {
                document.body.classList.add('keyboard-navigation');
            }
        });
        
        document.addEventListener('mousedown', () => {
            document.body.classList.remove('keyboard-navigation');
        });
    }
    
    setupSkipLink() {
        const skipLink = document.querySelector('.skip-link');
        skipLink?.addEventListener('click', (e) => {
            const target = document.querySelector(skipLink.getAttribute('href'));
            if (target) {
                target.setAttribute('tabindex', '-1');
                target.focus({ preventScroll: true });
            }
        });
    }
}

class Performance {
    constructor() {
        this.init();
    }
    
    init() {
        this.measureCoreWebVitals();
        this.lazyLoadImages();
        this.preloadCriticalResources();
    }
    
    measureCoreWebVitals() {
        if ('PerformanceObserver' in window) {
            try {
                const observer = new PerformanceObserver((list) => {
                    const entries = list.getEntries();
                    entries.forEach(entry => {
                        if (entry.entryType === 'paint') {
                            console.log(`${entry.name}: ${entry.startTime.toFixed(2)}ms`);
                        }
                    });
                });
                observer.observe({ type: 'paint', buffered: true });
            } catch (e) {
                console.log('Performance measurement not supported');
            }
        }
    }
    
    lazyLoadImages() {
        if ('IntersectionObserver' in window) {
            const imageObserver = new IntersectionObserver((entries) => {
                entries.forEach(entry => {
                    if (entry.isIntersecting) {
                        const img = entry.target;
                        if (img.dataset.src) {
                            img.src = img.dataset.src;
                            img.removeAttribute('data-src');
                        }
                        if (img.dataset.srcset) {
                            img.srcset = img.dataset.srcset;
                            img.removeAttribute('data-srcset');
                        }
                        imageObserver.unobserve(img);
                    }
                });
            }, {
                rootMargin: '50px'
            });
            
            document.querySelectorAll('img[data-src]').forEach(img => {
                imageObserver.observe(img);
            });
        }
    }
    
    preloadCriticalResources() {
        const criticalFonts = ['https://fonts.googleapis.com/css2?family=Noto+Sans+SC:wght@400;500;700&display=swap'];
        criticalFonts.forEach(font => {
            const link = document.createElement('link');
            link.rel = 'preload';
            link.as = 'style';
            link.href = font;
            document.head.appendChild(link);
        });
    }
}

document.addEventListener('DOMContentLoaded', () => {
    new Navigation();
    new DataTabs();
    new DownloadManager();
    new ContactForm();
    new Modal();
    new SmoothScroll();
    new Accessibility();
    new Performance();
});

if (module.hot) {
    module.hot.accept();
}
