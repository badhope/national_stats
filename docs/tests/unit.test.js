/**
 * Unit Tests for National Stats Website
 * Test Suite: Utils Module
 */

const Utils = {
    sanitize(str) {
        const div = document.createElement('div');
        div.textContent = str;
        return div.innerHTML;
    },
    
    escapeHtml(str) {
        const map = {
            '&': '&amp;',
            '<': '&lt;',
            '>': '&gt;',
            '"': '&quot;',
            "'": '&#039;'
        };
        return str.replace(/[&<>"']/g, m => map[m]);
    },
    
    debounce(func, wait) {
        let timeout;
        return function executedFunction(...args) {
            const later = () => {
                clearTimeout(timeout);
                func(...args);
            };
            clearTimeout(timeout);
            timeout = setTimeout(later, wait);
        };
    },
    
    throttle(func, limit) {
        let inThrottle;
        return function(...args) {
            if (!inThrottle) {
                func.apply(this, args);
                inThrottle = true;
                setTimeout(() => inThrottle = false, limit);
            }
        };
    }
};

function runUtilsTests() {
    const results = {
        passed: 0,
        failed: 0,
        tests: []
    };

    function test(name, fn) {
        try {
            fn();
            results.passed++;
            results.tests.push({ name, status: 'passed' });
            console.log(`✓ ${name}`);
        } catch (error) {
            results.failed++;
            results.tests.push({ name, status: 'failed', error: error.message });
            console.log(`✗ ${name}: ${error.message}`);
        }
    }

    function assertEqual(actual, expected, message = '') {
        if (actual !== expected) {
            throw new Error(`${message} Expected: ${expected}, Got: ${actual}`);
        }
    }

    function assertTrue(condition, message = '') {
        if (!condition) {
            throw new Error(message || 'Expected true');
        }
    }

    console.log('\n=== Utils Module Tests ===\n');

    test('sanitize: should escape HTML tags', () => {
        const result = Utils.sanitize('<script>alert("xss")</script>');
        assertEqual(result, '&lt;script&gt;alert(&quot;xss&quot;)&lt;/script&gt;');
    });

    test('sanitize: should escape ampersands', () => {
        const result = Utils.sanitize('A & B');
        assertEqual(result, 'A &amp; B');
    });

    test('sanitize: should handle empty string', () => {
        const result = Utils.sanitize('');
        assertEqual(result, '');
    });

    test('escapeHtml: should escape all HTML entities', () => {
        const result = Utils.escapeHtml('<>&"\'test');
        assertEqual(result, '&lt;&gt;&quot;&#039;test');
    });

    test('debounce: should delay function execution', async () => {
        let counter = 0;
        const debouncedFn = Utils.debounce(() => counter++, 100);
        
        debouncedFn();
        debouncedFn();
        debouncedFn();
        
        assertEqual(counter, 0, 'Should not execute immediately');
        
        await new Promise(r => setTimeout(r, 150));
        assertEqual(counter, 1, 'Should execute once after delay');
    });

    test('throttle: should limit function execution rate', async () => {
        let counter = 0;
        const throttledFn = Utils.throttle(() => counter++, 100);
        
        throttledFn();
        throttledFn();
        throttledFn();
        
        assertEqual(counter, 1, 'Should execute once immediately');
        
        await new Promise(r => setTimeout(r, 150));
        assertEqual(counter, 1, 'Should not execute again within throttle period');
        
        await new Promise(r => setTimeout(r, 100));
        throttledFn();
        assertEqual(counter, 2, 'Should execute again after throttle period');
    });

    console.log(`\n=== Results: ${results.passed}/${results.passed + results.failed} passed ===\n`);
    
    return results;
}

if (typeof module !== 'undefined' && module.exports) {
    module.exports = { runUtilsTests, Utils };
}

if (typeof window !== 'undefined') {
    window.runUtilsTests = runUtilsTests;
}
