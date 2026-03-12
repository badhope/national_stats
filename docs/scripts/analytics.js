/**
 * Google Analytics Integration Module
 * Supports both GA4 and Universal Analytics
 * 
 * Usage:
 *   Analytics.init('G-XXXXXXXXXX');  // GA4
 *   Analytics.init('UA-XXXXX-X');   // Universal Analytics
 *   Analytics.trackEvent('category', 'action', 'label', value);
 */

const Analytics = (function() {
    'use strict';

    const config = {
        trackingId: null,
        debug: false,
        anonymizeIp: true,
        respectDoNotTrack: true,
        sampleRate: 100,
        siteSpeedSampleRate: 1,
        userTimings: true,
        eventCategories: {
            NAVIGATION: 'Navigation',
            ENGAGEMENT: 'Engagement',
            FORM: 'Form',
            DOWNLOAD: 'Download',
            VIDEO: 'Video',
            ERROR: 'Error',
            PERFORMANCE: 'Performance'
        }
    };

    const state = {
        initialized: false,
        userId: null,
        sessionId: null,
        pageViews: 0
    };

    function log(message, data = {}) {
        if (config.debug) {
            console.log(`[Analytics] ${message}`, data);
        }
    }

    function shouldTrack() {
        if (config.respectDoNotTrack && navigator.doNotTrack === '1') {
            log('Do Not Track is enabled, not tracking');
            return false;
        }
        
        if (Math.random() * 100 > config.sampleRate) {
            log('User not in sample rate');
            return false;
        }
        
        return true;
    }

    function generateUserId() {
        const stored = localStorage.getItem('analytics_user_id');
        if (stored) return stored;
        
        const newId = 'xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx'.replace(/[xy]/g, function(c) {
            const r = Math.random() * 16 | 0;
            const v = c === 'x' ? r : (r & 0x3 | 0x8);
            return v.toString(16);
        });
        
        localStorage.setItem('analytics_user_id', newId);
        return newId;
    }

    function generateSessionId() {
        return `s_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
    }

    function isGA4(trackingId) {
        return trackingId.startsWith('G-');
    }

    function initGA4(trackingId) {
        log('Initializing GA4', { trackingId });
        
        window.dataLayer = window.dataLayer || [];
        window.gtag = function() {
            window.dataLayer.push(arguments);
        };
        window.gtag('js', new Date());
        window.gtag('config', trackingId, {
            'anonymize_ip': config.anonymizeIp,
            'sample_rate': config.sampleRate,
            'site_speed_sample_rate': config.siteSpeedSampleRate,
            'user_id': state.userId
        });
        
        const script = document.createElement('script');
        script.async = true;
        script.src = `https://www.googletagmanager.com/gtag/js?id=${trackingId}`;
        document.head.appendChild(script);
    }

    function initUniversalAnalytics(trackingId) {
        log('Initializing Universal Analytics', { trackingId });
        
        window.ga = window.ga || function() {
            (window.ga.q = window.ga.q || []).push(arguments);
        };
        window.ga.l = 1 * new Date();
        
        window.ga('create', trackingId, {
            'anonymizeIp': config.anonymizeIp,
            'sampleRate': config.sampleRate,
            'siteSpeedSampleRate': config.siteSpeedSampleRate,
            'userId': state.userId
        });
        
        window.ga('send', 'pageview');
        
        const script = document.createElement('script');
        script.async = true;
        script.src = `https://www.google-analytics.com/analytics.js`;
        document.head.appendChild(script);
    }

    function trackPageViewGA4(path, title) {
        if (!window.gtag) return;
        
        window.gtag('event', 'page_view', {
            page_path: path || window.location.pathname,
            page_title: title || document.title
        });
        
        log('Page view tracked (GA4)', { path, title });
    }

    function trackPageViewUA(path, title) {
        if (!window.ga) return;
        
        window.ga('send', 'pageview', {
            page: path || window.location.pathname,
            title: title || document.title
        });
        
        log('Page view tracked (UA)', { path, title });
    }

    function trackEventGA4(category, action, label, value) {
        if (!window.gtag) return;
        
        window.gtag('event', action, {
            event_category: category,
            event_label: label,
            value: value
        });
        
        log('Event tracked (GA4)', { category, action, label, value });
    }

    function trackEventUA(category, action, label, value) {
        if (!window.ga) return;
        
        window.ga('send', 'event', {
            eventCategory: category,
            eventAction: action,
            eventLabel: label,
            eventValue: value,
            nonInteraction: false
        });
        
        log('Event tracked (UA)', { category, action, label, value });
    }

    function trackTiming(category, variable, value, label) {
        if (isGA4(config.trackingId)) {
            trackEventGA4('timing', variable, label, value);
        } else {
            if (!window.ga) return;
            window.ga('send', 'timing', {
                timingCategory: category,
                timingVar: variable,
                timingValue: value,
                timingLabel: label
            });
        }
    }

    function trackException(description, fatal = false) {
        if (isGA4(config.trackingId)) {
            if (!window.gtag) return;
            window.gtag('event', 'exception', {
                description: description,
                fatal: fatal
            });
        } else {
            if (!window.ga) return;
            window.ga('send', 'exception', {
                exDescription: description,
                exFatal: fatal
            });
        }
        
        log('Exception tracked', { description, fatal });
    }

    function setUserProperties(properties) {
        if (isGA4(config.trackingId)) {
            if (!window.gtag) return;
            window.gtag('set', 'user_properties', properties);
        } else {
            if (!window.ga) return;
            window.ga('set', properties);
        }
        
        log('User properties set', properties);
    }

    function init(trackingId, options = {}) {
        if (state.initialized) {
            log('Already initialized');
            return;
        }
        
        if (!trackingId) {
            log('No tracking ID provided');
            return;
        }
        
        Object.assign(config, options);
        
        if (!shouldTrack()) {
            log('Tracking disabled');
            return;
        }
        
        config.trackingId = trackingId;
        state.userId = generateUserId();
        state.sessionId = generateSessionId();
        
        if (isGA4(trackingId)) {
            initGA4(trackingId);
        } else {
            initUniversalAnalytics(trackingId);
        }
        
        state.initialized = true;
        log('Analytics initialized', { trackingId, userId: state.userId });
    }

    function trackPageView(path, title) {
        if (!state.initialized) return;
        
        state.pageViews++;
        
        if (isGA4(config.trackingId)) {
            trackPageViewGA4(path, title);
        } else {
            trackPageViewUA(path, title);
        }
    }

    function trackEvent(category, action, label = '', value = 0) {
        if (!state.initialized || !shouldTrack()) return;
        
        if (isGA4(config.trackingId)) {
            trackEventGA4(category, action, label, value);
        } else {
            trackEventUA(category, action, label, value);
        }
    }

    function trackCustomEvent(eventName, params = {}) {
        if (!state.initialized || !window.gtag) return;
        
        window.gtag('event', eventName, params);
        log('Custom event tracked', { eventName, params });
    }

    return {
        init,
        config,
        trackPageView,
        trackEvent,
        trackTiming,
        trackException,
        trackCustomEvent,
        setUserProperties,
        isGA4
    };
})();

if (typeof module !== 'undefined' && module.exports) {
    module.exports = Analytics;
}

if (typeof window !== 'undefined') {
    window.Analytics = Analytics;
}
