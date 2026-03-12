class Navigation {
    constructor() {
        this.header = document.querySelector('.header');
        this.navToggle = document.querySelector('.header__toggle');
        this.nav = document.querySelector('.nav');
        this.navLinks = document.querySelectorAll('.nav__link');
        this.backToTop = document.querySelector('.back-to-top');
        
        this.init();
    }
    
    init() {
        this.setupMobileMenu();
        this.setupBackToTop();
        this.setupScrollSpy();
    }
    
    setupMobileMenu() {
        if (!this.navToggle || !this.nav) return;
        
        this.navToggle.addEventListener('click', () => {
            const isOpen = this.nav.classList.toggle('nav--open');
            this.navToggle.setAttribute('aria-expanded', isOpen);
            
            if (isOpen) {
                this.nav.querySelector('.nav__link').focus();
            }
        });
        
        document.addEventListener('click', (e) => {
            if (!this.nav.contains(e.target) && !this.navToggle.contains(e.target)) {
                this.nav.classList.remove('nav--open');
                this.navToggle.setAttribute('aria-expanded', 'false');
            }
        });
        
        this.navLinks.forEach(link => {
            link.addEventListener('keydown', (e) => {
                if (e.key === 'Escape') {
                    this.nav.classList.remove('nav--open');
                    this.navToggle.setAttribute('aria-expanded', 'false');
                    this.navToggle.focus();
                }
            });
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
        
        window.addEventListener('scroll', () => {
            requestAnimationFrame(toggleVisibility);
        }, { passive: true });
        
        this.backToTop.addEventListener('click', () => {
            window.scrollTo({
                top: 0,
                behavior: 'smooth'
            });
        });
        
        this.backToTop.addEventListener('keydown', (e) => {
            if (e.key === 'Enter' || e.key === ' ') {
                e.preventDefault();
                this.backToTop.click();
            }
        });
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
            rootMargin: '-20% 0px -60% 0px'
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
            this.showToast('下载成功！', 'success');
            
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
        } else {
            button.classList.remove('download-btn--loading');
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
        
        setTimeout(() => {
            this.resetButton(button);
        }, 3000);
    }
    
    setButtonError(button) {
        button.classList.remove('download-btn--loading');
        button.classList.add('download-btn--error');
        
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
        `;
        
        this.toastContainer.appendChild(toast);
        
        setTimeout(() => {
            toast.classList.add('toast--hiding');
            toast.addEventListener('animationend', () => {
                toast.remove();
            });
        }, 3000);
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
            
            const headerHeight = document.querySelector('.header')?.offsetHeight || 0;
            const targetPosition = target.getBoundingClientRect().top + window.scrollY - headerHeight;
            
            window.scrollTo({
                top: targetPosition,
                behavior: 'smooth'
            });
            
            target.setAttribute('tabindex', '-1');
            target.focus({ preventScroll: true });
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
    }
    
    announcePageLoad() {
        const main = document.querySelector('main');
        if (main) {
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
}

class Performance {
    constructor() {
        this.init();
    }
    
    init() {
        this.measureFCP();
        this.lazyLoadImages();
    }
    
    measureFCP() {
        if ('PerformanceObserver' in window) {
            try {
                const observer = new PerformanceObserver((list) => {
                    const entries = list.getEntries();
                    const fcpEntry = entries.find(entry => entry.name === 'first-contentful-paint');
                    if (fcpEntry) {
                        console.log(`FCP: ${fcpEntry.startTime.toFixed(2)}ms`);
                        observer.disconnect();
                    }
                });
                observer.observe({ type: 'paint', buffered: true });
            } catch (e) {
                console.log('FCP measurement not supported');
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
                        imageObserver.unobserve(img);
                    }
                });
            });
            
            document.querySelectorAll('img[data-src]').forEach(img => {
                imageObserver.observe(img);
            });
        }
    }
}

document.addEventListener('DOMContentLoaded', () => {
    new Navigation();
    new DownloadManager();
    new SmoothScroll();
    new Accessibility();
    new Performance();
});

if (module.hot) {
    module.hot.accept();
}
