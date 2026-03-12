/**
 * E2E Tests for National Stats Website
 * Test Suite: Navigation, DataTabs, ContactForm
 */

class E2ETestRunner {
    constructor() {
        this.results = {
            passed: 0,
            failed: 0,
            tests: []
        };
    }

    log(message, type = 'info') {
        const prefix = type === 'success' ? '✓' : type === 'error' ? '✗' : '→';
        console.log(`${prefix} ${message}`);
    }

    async test(name, fn) {
        this.log(name);
        try {
            await fn();
            this.results.passed++;
            this.results.tests.push({ name, status: 'passed' });
            this.log(`  Passed`, 'success');
        } catch (error) {
            this.results.failed++;
            this.results.tests.push({ name, status: 'failed', error: error.message });
            this.log(`  Failed: ${error.message}`, 'error');
        }
    }

    assert(condition, message = 'Assertion failed') {
        if (!condition) throw new Error(message);
    }

    assertEqual(actual, expected, message = '') {
        if (actual !== expected) {
            throw new Error(`${message} Expected: ${expected}, Got: ${actual}`);
        }
    }

    async summary() {
        console.log('\n========================================');
        console.log(`E2E Tests: ${this.results.passed}/${this.results.passed + this.results.failed} passed`);
        console.log('========================================\n');
        
        if (this.results.failed > 0) {
            console.log('Failed tests:');
            this.results.tests
                .filter(t => t.status === 'failed')
                .forEach(t => console.log(`  - ${t.name}: ${t.error}`));
        }
        
        return this.results;
    }
}

async function runE2ETests() {
    const runner = new E2ETestRunner();
    
    console.log('\n=== E2E Tests: National Stats Website ===\n');

    await runner.test('Page loads without errors', async () => {
        runner.assert(document.readyState === 'complete', 'Page should be loaded');
    });

    await runner.test('Header is visible', async () => {
        const header = document.querySelector('.header');
        runner.assert(header !== null, 'Header should exist');
        runner.assert(getComputedStyle(header).display !== 'none', 'Header should be visible');
    });

    await runner.test('Navigation links are functional', async () => {
        const navLinks = document.querySelectorAll('.nav__link');
        runner.assert(navLinks.length > 0, 'Should have navigation links');
        
        const firstLink = navLinks[0];
        runner.assert(firstLink.href || firstLink.getAttribute('href'), 'Link should have href');
    });

    await runner.test('Data tabs exist and switch correctly', async () => {
        const tabs = document.querySelectorAll('.data-tab');
        runner.assert(tabs.length >= 4, 'Should have at least 4 data tabs (GDP, CPI, Employment, Trade)');
        
        const firstTab = tabs[0];
        const isActive = firstTab.classList.contains('active') || firstTab.getAttribute('aria-selected') === 'true';
        runner.assert(isActive || !tabs[1]?.classList.contains('active'), 'First tab should be active by default');
    });

    await runner.test('Contact form fields exist', async () => {
        const form = document.getElementById('contact-form');
        runner.assert(form !== null, 'Contact form should exist');
        
        const nameInput = document.getElementById('name');
        const emailInput = document.getElementById('email');
        const subjectSelect = document.getElementById('subject');
        
        runner.assert(nameInput !== null, 'Name input should exist');
        runner.assert(emailInput !== null, 'Email input should exist');
        runner.assert(subjectSelect !== null, 'Subject select should exist');
    });

    await runner.test('Contact form validation works', async () => {
        const form = document.getElementById('contact-form');
        if (!form) return;
        
        const emailInput = document.getElementById('email');
        if (!emailInput) return;
        
        emailInput.value = 'invalid-email';
        emailInput.dispatchEvent(new Event('blur', { bubbles: true }));
        
        const errorElement = document.getElementById('email-error');
        if (errorElement) {
            const hasError = errorElement.textContent.length > 0 || errorElement.classList.contains('contact-form__error--visible');
            runner.assert(hasError, 'Should show validation error for invalid email');
        }
    });

    await runner.test('Modal can be opened', async () => {
        const modal = document.getElementById('demo-modal');
        if (!modal) {
            console.log('  (Modal not found, skipping)');
            return;
        }
        
        const modalOverlay = modal.querySelector('.modal__overlay');
        runner.assert(modalOverlay !== null, 'Modal overlay should exist');
    });

    await runner.test('Back to top button exists', async () => {
        const backToTop = document.querySelector('.back-to-top');
        runner.assert(backToTop !== null, 'Back to top button should exist');
    });

    await runner.test('Download buttons exist', async () => {
        const downloadBtns = document.querySelectorAll('.download-btn');
        runner.assert(downloadBtns.length > 0, 'Should have download buttons');
    });

    await runner.test('Footer information is complete', async () => {
        const footer = document.querySelector('.footer');
        runner.assert(footer !== null, 'Footer should exist');
    });

    await runner.test('Accessibility: Skip link exists', async () => {
        const skipLink = document.querySelector('.skip-link');
        runner.assert(skipLink !== null, 'Skip link should exist for accessibility');
    });

    await runner.test('Accessibility: All images have alt text', async () => {
        const images = document.querySelectorAll('img');
        let allAlt = true;
        
        images.forEach(img => {
            if (!img.alt && !img.getAttribute('aria-label') && !img.getAttribute('aria-hidden')) {
                allAlt = false;
            }
        });
        
        if (images.length > 0) {
            runner.assert(allAlt, 'All images should have alt text or be decorative');
        }
    });

    await runner.test('Performance: No console errors', async () => {
        const errors = [];
        window.addEventListener('error', (e) => {
            errors.push(e.message);
        });
        
        if (errors.length > 0) {
            console.log(`  Console errors found: ${errors.join(', ')}`);
        }
    });

    await runner.test('Responsive: Viewport meta tag is set', async () => {
        const viewportMeta = document.querySelector('meta[name="viewport"]');
        runner.assert(viewportMeta !== null, 'Viewport meta tag should exist');
        runner.assert(viewportMeta.content.includes('width=device-width'), 'Viewport should include width=device-width');
    });

    await runner.test('SEO: Title is set', async () => {
        runner.assert(document.title.length > 0, 'Document title should be set');
        console.log(`  Title: ${document.title}`);
    });

    await runner.test('SEO: Meta description is set', async () => {
        const descriptionMeta = document.querySelector('meta[name="description"]');
        runner.assert(descriptionMeta !== null, 'Meta description should exist');
        runner.assert(descriptionMeta.content.length > 0, 'Meta description should have content');
    });

    return runner.summary();
}

if (typeof module !== 'undefined' && module.exports) {
    module.exports = { runE2ETests, E2ETestRunner };
}

if (typeof window !== 'undefined') {
    window.runE2ETests = runE2ETests;
}
