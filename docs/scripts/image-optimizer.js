/**
 * Image Optimization Module
 * Provides WebP conversion, responsive images, and lazy loading
 */

const ImageOptimizer = (function() {
    'use strict';

    const config = {
        quality: 80,
        maxWidth: 1920,
        breakpoints: [320, 640, 768, 1024, 1280, 1920],
        lazyLoadRootMargin: '50px',
        lazyLoadThreshold: 0.1
    };

    function supportsWebP() {
        const canvas = document.createElement('canvas');
        canvas.width = 1;
        canvas.height = 1;
        return canvas.toDataURL('image/webp').startsWith('data:image/webp');
    }

    function supportsSrcSet() {
        const img = document.createElement('img');
        return 'srcset' in img;
    }

    function generateSrcSet(imagePath, breakpoints = config.breakpoints) {
        const ext = imagePath.split('.').pop();
        const basePath = imagePath.replace(`.${ext}`, '');
        
        return breakpoints.map(width => {
            const newPath = `${basePath}-${width}w.${ext}`;
            return `${newPath} ${width}w`;
        }).join(', ');
    }

    function createPictureElement(imageConfig) {
        const {
            src,
            alt,
            className = '',
            lazy = true,
            sizes = '100vw',
            sources = []
        } = imageConfig;

        const picture = document.createElement('picture');
        picture.className = className;

        sources.forEach(source => {
            const sourceEl = document.createElement('source');
            sourceEl.srcset = source.srcset;
            sourceEl.type = source.type;
            sourceEl.media = source.media;
            picture.appendChild(sourceEl);
        });

        const img = document.createElement('img');
        img.src = src;
        img.alt = alt;
        img.loading = lazy ? 'lazy' : 'eager';
        
        if (sizes) {
            img.sizes = sizes;
        }

        if (lazy) {
            img.classList.add('lazy');
        }

        picture.appendChild(img);
        return picture;
    }

    function initLazyLoading() {
        if (!('IntersectionObserver' in window)) {
            document.querySelectorAll('img[data-src]').forEach(img => {
                if (img.dataset.src) {
                    img.src = img.dataset.src;
                    img.removeAttribute('data-src');
                }
            });
            return;
        }

        const imageObserver = new IntersectionObserver((entries, observer) => {
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
                    
                    img.classList.add('loaded');
                    img.classList.remove('lazy');
                    observer.unobserve(img);
                }
            });
        }, {
            rootMargin: config.lazyLoadRootMargin,
            threshold: config.lazyLoadThreshold
        });

        document.querySelectorAll('img.lazy, img[data-src]').forEach(img => {
            imageObserver.observe(img);
        });
    }

    function initResponsiveImages() {
        document.querySelectorAll('img[data-responsive]').forEach(img => {
            const src = img.dataset.src || img.src;
            const breakpoints = img.dataset.breakpoints 
                ? img.dataset.breakpoints.split(',').map(Number) 
                : config.breakpoints;
            
            const srcset = generateSrcSet(src, breakpoints);
            img.srcset = srcset;
            img.sizes = img.dataset.sizes || '(max-width: 768px) 100vw, 50vw';
            img.removeAttribute('data-responsive');
        });
    }

    function convertToWebP(imageUrl, quality = config.quality) {
        return new Promise((resolve, reject) => {
            const img = new Image();
            img.crossOrigin = 'Anonymous';
            
            img.onload = () => {
                const canvas = document.createElement('canvas');
                canvas.width = img.naturalWidth;
                canvas.height = img.naturalHeight;
                
                const ctx = canvas.getContext('2d');
                ctx.drawImage(img, 0, 0);
                
                canvas.toBlob(
                    (blob) => {
                        if (blob) {
                            resolve(URL.createObjectURL(blob));
                        } else {
                            reject(new Error('WebP conversion failed'));
                        }
                    },
                    'image/webp',
                    quality / 100
                );
            };
            
            img.onerror = () => reject(new Error('Failed to load image'));
            img.src = imageUrl;
        });
    }

    function getPlaceholderDataUrl(width, height, color = '#e2e8f0') {
        const svg = `<svg xmlns="http://www.w3.org/2000/svg" width="${width}" height="${height}"><rect fill="${color}" width="100%" height="100%"/></svg>`;
        return `data:image/svg+xml,${encodeURIComponent(svg)}`;
    }

    function init() {
        console.log('ImageOptimizer initialized');
        
        if (!supportsWebP()) {
            console.warn('WebP not supported, using fallback images');
        }
        
        initLazyLoading();
        initResponsiveImages();
    }

    return {
        init,
        config,
        supportsWebP,
        supportsSrcSet,
        generateSrcSet,
        createPictureElement,
        convertToWebP,
        getPlaceholderDataUrl,
        initLazyLoading,
        initResponsiveImages
    };
})();

if (typeof module !== 'undefined' && module.exports) {
    module.exports = ImageOptimizer;
}

if (typeof window !== 'undefined') {
    window.ImageOptimizer = ImageOptimizer;
}
