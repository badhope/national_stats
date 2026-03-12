/**
 * Real User Monitoring (RUM) Module
 * Tracks Core Web Vitals and custom events
 */

const RumMonitor = (function() {
    'use strict';

    const config = {
        endpoint: null,
        sampleRate: 100,
        debug: false,
        metrics: {
            fcp: true,
            lcp: true,
            cls: true,
            inp: true,
            ttfb: true
        }
    };

    const metrics = {
        fcp: null,
        lcp: null,
        cls: null,
        inp: null,
        ttfb: null
    };

    const events = [];

    function log(message, data = {}) {
        if (config.debug) {
            console.log(`[RUM] ${message}`, data);
        }
    }

    function shouldSample() {
        return Math.random() * 100 < config.sampleRate;
    }

    function getNavigationTiming() {
        if (!performance || !performance.timing) return null;
        
        const timing = performance.timing;
        const navStart = timing.navigationStart;
        
        return {
            dns: timing.domainLookupEnd - timing.domainLookupStart,
            tcp: timing.connectEnd - timing.connectStart,
            ssl: timing.secureConnectionStart > 0 
                ? timing.connectEnd - timing.secureConnectionStart 
                : 0,
            ttfb: timing.responseStart - timing.requestStart,
            download: timing.responseEnd - timing.responseStart,
            domReady: timing.domContentLoadedEventEnd - navStart,
            loadComplete: timing.loadEventEnd - navStart
        };
    }

    function getResourceTiming() {
        if (!performance || !performance.getEntriesByType) return [];
        
        return performance.getEntriesByType('resource')
            .filter(entry => entry.initiatorType !== 'navigation')
            .slice(0, 10)
            .map(entry => ({
                name: entry.name,
                type: entry.initiatorType,
                duration: entry.duration.toFixed(2),
                size: entry.transferSize || 0
            }));
    }

    function measureFCP() {
        if (!('PerformanceObserver' in window)) return;
        
        try {
            const observer = new PerformanceObserver((list) => {
                const entries = list.getEntries();
                const fcpEntry = entries.find(e => e.name === 'first-contentful-paint');
                
                if (fcpEntry) {
                    metrics.fcp = Math.round(fcpEntry.startTime);
                    log('FCP measured', { fcp: metrics.fcp });
                    sendMetric('fcp', metrics.fcp);
                }
            });
            
            observer.observe({ type: 'paint', buffered: true });
        } catch (e) {
            log('FCP measurement failed', { error: e.message });
        }
    }

    function measureLCP() {
        if (!('PerformanceObserver' in window)) return;
        
        try {
            const observer = new PerformanceObserver((list) => {
                const entries = list.getEntries();
                const lastEntry = entries[entries.length - 1];
                
                if (lastEntry) {
                    metrics.lcp = Math.round(lastEntry.startTime);
                    log('LCP measured', { lcp: metrics.lcp });
                    sendMetric('lcp', metrics.lcp);
                }
            });
            
            observer.observe({ type: 'largest-contentful-paint', buffered: true });
        } catch (e) {
            log('LCP measurement failed', { error: e.message });
        }
    }

    function measureCLS() {
        if (!('PerformanceObserver' in window)) return;
        
        try {
            let clsValue = 0;
            
            const observer = new PerformanceObserver((list) => {
                for (const entry of list.getEntries()) {
                    if (!entry.hadRecentInput) {
                        clsValue += entry.value;
                    }
                }
                
                metrics.cls = clsValue.toFixed(4);
                log('CLS measured', { cls: metrics.cls });
                sendMetric('cls', parseFloat(metrics.cls));
            });
            
            observer.observe({ type: 'layout-shift', buffered: true });
        } catch (e) {
            log('CLS measurement failed', { error: e.message });
        }
    }

    function measureINP() {
        if (!('PerformanceObserver' in window)) return;
        
        try {
            let maxDuration = 0;
            let maxEntry = null;
            
            const observer = new PerformanceObserver((list) => {
                for (const entry of list.getEntries()) {
                    if (entry.interactionId && entry.duration > maxDuration) {
                        maxDuration = entry.duration;
                        maxEntry = entry;
                    }
                }
                
                if (maxEntry) {
                    metrics.inp = Math.round(maxDuration);
                    log('INP measured', { inp: metrics.inp });
                    sendMetric('inp', metrics.inp);
                }
            });
            
            observer.observe({ type: 'event', buffered: true, durationThreshold: 16 });
        } catch (e) {
            log('INP measurement failed', { error: e.message });
        }
    }

    function measureTTFB() {
        if (!performance || !performance.timing) return;
        
        const timing = performance.timing;
        const ttfb = timing.responseStart - timing.navigationStart;
        
        if (ttfb > 0) {
            metrics.ttfb = ttfb;
            log('TTFB measured', { ttfb: metrics.ttfb });
            sendMetric('ttfb', metrics.ttfb);
        }
    }

    function sendMetric(name, value) {
        if (!config.endpoint) {
            log(`Metric: ${name}`, { value });
            return;
        }
        
        if (!shouldSample()) return;
        
        const payload = {
            name,
            value,
            url: window.location.href,
            timestamp: Date.now(),
            userAgent: navigator.userAgent,
            viewport: {
                width: window.innerWidth,
                height: window.innerHeight
            },
            connection: navigator.connection 
                ? navigator.connection.effectiveType 
                : 'unknown'
        };
        
        navigator.sendBeacon 
            ? navigator.sendBeacon(config.endpoint, JSON.stringify(payload))
            : fetch(config.endpoint, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(payload),
                keepalive: true
            }).catch(() => {});
    }

    function trackEvent(category, action, label = '', value = 0) {
        const event = {
            category,
            action,
            label,
            value,
            timestamp: Date.now(),
            url: window.location.href
        };
        
        events.push(event);
        log('Event tracked', event);
        
        if (config.endpoint) {
            sendMetric('event', { ...event, type: 'custom' });
        }
    }

    function trackError(error, context = {}) {
        const errorData = {
            message: error.message || String(error),
            stack: error.stack || '',
            ...context,
            timestamp: Date.now(),
            url: window.location.href
        };
        
        log('Error tracked', errorData);
        
        if (config.endpoint) {
            sendMetric('error', errorData);
        }
    }

    function getMetrics() {
        return {
            ...metrics,
            navigation: getNavigationTiming(),
            resources: getResourceTiming()
        };
    }

    function getEvents() {
        return [...events];
    }

    function init(options = {}) {
        Object.assign(config, options);
        
        if (!shouldSample() && !config.debug) {
            log('User not sampled, RUM disabled');
            return;
        }
        
        log('Initializing RUM Monitor', config);
        
        if (config.metrics.fcp) measureFCP();
        if (config.metrics.lcp) measureLCP();
        if (config.metrics.cls) measureCLS();
        if (config.metrics.inp) measureINP();
        if (config.metrics.ttfb) measureTTFB();
        
        window.addEventListener('error', (e) => {
            trackError(e.error || e.message, { type: 'window_error' });
        });
        
        window.addEventListener('unhandledrejection', (e) => {
            trackError(e.reason, { type: 'unhandled_rejection' });
        });
        
        log('RUM Monitor initialized');
    }

    return {
        init,
        config,
        getMetrics,
        getEvents,
        trackEvent,
        trackError,
        sendMetric
    };
})();

if (typeof module !== 'undefined' && module.exports) {
    module.exports = RumMonitor;
}

if (typeof window !== 'undefined') {
    window.RumMonitor = RumMonitor;
}
