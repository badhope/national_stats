/**
 * Internationalization (i18n) Module
 * Supports Chinese (Simplified) and English
 * 
 * Usage:
 *   i18n.init('zh-CN');  // Initialize with Chinese
 *   i18n.t('key');       // Get translated string
 *   i18n.setLocale('en'); // Switch to English
 */

const i18n = (function() {
    'use strict';

    const config = {
        defaultLocale: 'zh-CN',
        currentLocale: 'zh-CN',
        fallbackLocale: 'zh-CN',
        storageKey: 'i18n_locale',
        autoDetect: true,
        debug: false
    };

    const translations = {
        'zh-CN': {
            nav: {
                home: '首页',
                analysis: '数据分析',
                data: '数据中心',
                reports: '资源下载',
                contact: '联系我们'
            },
            hero: {
                title: '宏观经济数据分析平台',
                subtitle: '权威数据来源，深度分析洞察',
                cta: '开始探索',
                secondary: '查看演示'
            },
            features: {
                title: '核心功能',
                subtitle: '强大而专业的经济数据分析工具集'
            },
            data: {
                title: '数据中心',
                subtitle: '权威数据来源，全面经济指标覆盖',
                gdp: '国内生产总值',
                cpi: '居民消费价格指数',
                employment: '就业形势',
                trade: '对外贸易'
            },
            download: {
                title: '资源下载',
                subtitle: '获取最新数据和分析报告',
                button: '立即下载'
            },
            contact: {
                title: '联系我们',
                subtitle: '有任何问题？我们的团队随时为您提供帮助',
                name: '姓名',
                email: '邮箱',
                subject: '主题',
                message: '留言',
                privacy: '我已阅读并同意隐私政策',
                submit: '提交咨询'
            },
            footer: {
                description: '专业的宏观经济数据分析平台',
                copyright: '© 2024 National Stats. All rights reserved.',
                links: '快速链接',
                contact: '联系方式',
                follow: '关注我们'
            },
            common: {
                loading: '加载中...',
                error: '出错了',
                retry: '重试',
                close: '关闭',
                cancel: '取消',
                confirm: '确认',
                success: '成功',
                learnMore: '了解更多'
            }
        },
        'en': {
            nav: {
                home: 'Home',
                analysis: 'Analysis',
                data: 'Data Center',
                reports: 'Downloads',
                contact: 'Contact'
            },
            hero: {
                title: 'Macroeconomic Data Analysis Platform',
                subtitle: 'Authoritative data sources, deep analytical insights',
                cta: 'Explore Now',
                secondary: 'View Demo'
            },
            features: {
                title: 'Core Features',
                subtitle: 'Powerful and professional economic data analysis tools'
            },
            data: {
                title: 'Data Center',
                subtitle: 'Authoritative data sources, comprehensive economic indicators',
                gdp: 'GDP',
                cpi: 'CPI',
                employment: 'Employment',
                trade: 'Trade'
            },
            download: {
                title: 'Downloads',
                subtitle: 'Get the latest data and analysis reports',
                button: 'Download Now'
            },
            contact: {
                title: 'Contact Us',
                subtitle: 'Have questions? Our team is here to help',
                name: 'Name',
                email: 'Email',
                subject: 'Subject',
                message: 'Message',
                privacy: 'I have read and agree to the privacy policy',
                submit: 'Submit'
            },
            footer: {
                description: 'Professional macroeconomic data analysis platform',
                copyright: '© 2024 National Stats. All rights reserved.',
                links: 'Quick Links',
                contact: 'Contact',
                follow: 'Follow Us'
            },
            common: {
                loading: 'Loading...',
                error: 'Error occurred',
                retry: 'Retry',
                close: 'Close',
                cancel: 'Cancel',
                confirm: 'Confirm',
                success: 'Success',
                learnMore: 'Learn More'
            }
        }
    };

    function log(message, data = {}) {
        if (config.debug) {
            console.log(`[i18n] ${message}`, data);
        }
    }

    function getNestedValue(obj, path) {
        return path.split('.').reduce((acc, key) => acc && acc[key], obj);
    }

    function translate(key, params = {}) {
        const locale = config.currentLocale;
        const fallback = config.fallbackLocale;
        
        let value = getNestedValue(translations[locale], key);
        
        if (value === undefined && locale !== fallback) {
            value = getNestedValue(translations[fallback], key);
        }
        
        if (value === undefined) {
            log(`Translation not found: ${key}`);
            return key;
        }
        
        if (typeof value === 'string' && Object.keys(params).length > 0) {
            Object.entries(params).forEach(([paramKey, paramValue]) => {
                value = value.replace(new RegExp(`{${paramKey}}`, 'g'), paramValue);
            });
        }
        
        return value;
    }

    function setLocale(locale) {
        if (!translations[locale]) {
            log(`Locale not supported: ${locale}`);
            return false;
        }
        
        config.currentLocale = locale;
        
        try {
            localStorage.setItem(config.storageKey, locale);
        } catch (e) {
            log('Failed to save locale to storage', { error: e.message });
        }
        
        log('Locale changed', { locale });
        
        document.dispatchEvent(new CustomEvent('i18n:localeChanged', {
            detail: { locale }
        }));
        
        return true;
    }

    function getLocale() {
        return config.currentLocale;
    }

    function getSupportedLocales() {
        return Object.keys(translations);
    }

    function detectLocale() {
        const stored = localStorage.getItem(config.storageKey);
        if (stored && translations[stored]) {
            return stored;
        }
        
        const browserLang = navigator.language || navigator.userLanguage;
        if (translations[browserLang]) {
            return browserLang;
        }
        
        const shortLang = browserLang.split('-')[0];
        const matchedLocale = Object.keys(translations).find(locale => 
            locale.startsWith(shortLang)
        );
        
        return matchedLocale || config.defaultLocale;
    }

    function updatePageContent() {
        document.querySelectorAll('[data-i18n]').forEach(element => {
            const key = element.getAttribute('data-i18n');
            const translation = translate(key);
            
            if (element.tagName === 'INPUT' && element.placeholder) {
                element.placeholder = translation;
            } else {
                element.textContent = translation;
            }
        });
        
        document.querySelectorAll('[data-i18n-html]').forEach(element => {
            const key = element.getAttribute('data-i18n-html');
            element.innerHTML = translate(key);
        });
        
        document.querySelectorAll('[data-i18n-attr]').forEach(element => {
            const attrData = element.getAttribute('data-i18n-attr');
            const [attr, key] = attrData.split(':');
            if (attr && key) {
                element.setAttribute(attr, translate(key));
            }
        });
    }

    function init(options = {}) {
        Object.assign(config, options);
        
        if (config.autoDetect) {
            const detected = detectLocale();
            config.currentLocale = detected;
        }
        
        log('Initialized', { 
            locale: config.currentLocale, 
            default: config.defaultLocale 
        });
        
        updatePageContent();
        
        document.addEventListener('click', (e) => {
            const langSwitch = e.target.closest('[data-lang-switch]');
            if (langSwitch) {
                const locale = langSwitch.getAttribute('data-lang-switch');
                setLocale(locale);
                updatePageContent();
            }
        });
        
        return this;
    }

    return {
        init,
        config,
        t: translate,
        setLocale,
        getLocale,
        getSupportedLocales,
        updatePageContent
    };
})();

if (typeof module !== 'undefined' && module.exports) {
    module.exports = i18n;
}

if (typeof window !== 'undefined') {
    window.i18n = i18n;
}
