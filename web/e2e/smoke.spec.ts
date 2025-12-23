import { test, expect, FIXTURES } from './fixtures';

test.describe('Smoke Tests', () => {
	test.describe('Application Loading', () => {
		test('should load the application and show drop zone', async ({ page }) => {
			await page.goto('/');
			await expect(page.getByTestId('app-container')).toBeVisible();
			await expect(page.getByTestId('drop-zone')).toBeVisible();
			await expect(page.getByTestId('select-files-button')).toBeVisible();
			await expect(page.getByTestId('main-content')).toHaveAttribute('data-view-mode', 'drop');
		});

		test('should initialize WASM successfully', async ({ page, waitForWasm }) => {
			await page.goto('/');
			const wasmLoading = page.getByTestId('wasm-loading');
			const controlsFooter = page.getByTestId('controls-footer');
			await waitForWasm();
			await expect(wasmLoading).not.toBeVisible();
			await expect(controlsFooter).toHaveAttribute('data-wasm-ready', 'true');
		});
	});

	test.describe('PNG Compression', () => {
		test('should upload and compress a PNG file', async ({ page, waitForWasm, uploadAndWaitForCompression }) => {
			await page.goto('/');
			await waitForWasm();
			await uploadAndWaitForCompression(FIXTURES.PNG);
			await expect(page.getByTestId('main-content')).toHaveAttribute('data-view-mode', 'single');
			await expect(page.getByTestId('single-view')).toBeVisible();
			await expect(page.getByTestId('image-name')).toContainText('playground.png');
			await expect(page.getByTestId('image-dimensions')).toBeVisible();
			await expect(page.getByTestId('compressed-image-overlay')).toBeVisible();
			await expect(page.getByTestId('comparison-slider')).toBeVisible();
		});

		test('should show compression stats for PNG', async ({ page, waitForWasm, uploadAndWaitForCompression }) => {
			await page.goto('/');
			await waitForWasm();
			await uploadAndWaitForCompression(FIXTURES.PNG);
			await expect(page.getByTestId('total-original-size')).toBeVisible();
			await expect(page.getByTestId('total-compressed-size')).toBeVisible();
			await expect(page.getByTestId('total-savings')).toBeVisible();
			const savingsText = await page.getByTestId('total-savings').textContent();
			expect(savingsText).toMatch(/^[+-][\d.]+%$/);
		});

		test('should allow changing PNG compression settings', async ({ page, waitForWasm, uploadAndWaitForCompression }) => {
			await page.goto('/');
			await waitForWasm();
			await uploadAndWaitForCompression(FIXTURES.PNG);
			await expect(page.getByTestId('png-preset-slider')).toBeVisible();
			const initialSize = await page.getByTestId('total-compressed-size').textContent();
			// Change from default (1=Auto) to smaller (0)
			await page.getByTestId('png-preset-slider').fill('0');
			await page.getByTestId('png-preset-slider').dispatchEvent('change');
			await expect(page.getByTestId('download-button')).toBeVisible({ timeout: 60000 });
			await expect(page.getByTestId('total-compressed-size')).toBeVisible();
		});
	});

	test.describe('JPEG Compression', () => {
		test('should upload and compress a JPEG file', async ({ page, waitForWasm, uploadAndWaitForCompression }) => {
			await page.goto('/');
			await waitForWasm();
			await uploadAndWaitForCompression(FIXTURES.JPEG);
			await expect(page.getByTestId('main-content')).toHaveAttribute('data-view-mode', 'single');
			await expect(page.getByTestId('single-view')).toBeVisible();
			await expect(page.getByTestId('image-name')).toContainText('multi-agent.jpg');
		});

		test('should show JPEG quality controls for JPEG format', async ({ page, waitForWasm, uploadAndWaitForCompression }) => {
			await page.goto('/');
			await waitForWasm();
			await uploadAndWaitForCompression(FIXTURES.JPEG);
			await expect(page.getByTestId('quality-slider')).toBeVisible();
			await expect(page.getByTestId('quality-value')).toBeVisible();
			// PNG preset slider should not be visible for JPEG
			await expect(page.getByTestId('png-preset-slider')).not.toBeVisible();
		});
	});

	test.describe('Download', () => {
		test('should enable download button after compression', async ({ page, waitForWasm, uploadAndWaitForCompression }) => {
			await page.goto('/');
			await waitForWasm();
			await uploadAndWaitForCompression(FIXTURES.PNG);
			const downloadButton = page.getByTestId('download-button');
			await expect(downloadButton).toBeVisible();
			await expect(downloadButton).toBeEnabled();
			await expect(downloadButton).toHaveText('Download');
		});

		test('should trigger download when clicking download button', async ({ page, waitForWasm, uploadAndWaitForCompression }) => {
			await page.goto('/');
			await waitForWasm();
			await uploadAndWaitForCompression(FIXTURES.PNG);
			const downloadPromise = page.waitForEvent('download', { timeout: 10000 });
			await page.getByTestId('download-button').click();
			const download = await downloadPromise;
			expect(download.suggestedFilename()).toContain('playground');
			expect(download.suggestedFilename()).toContain('-compressed');
		});
	});

	test.describe('Image Comparison Slider', () => {
		test('should have working comparison slider', async ({ page, waitForWasm, uploadAndWaitForCompression }) => {
			await page.goto('/');
			await waitForWasm();
			await uploadAndWaitForCompression(FIXTURES.PNG);
			const slider = page.getByTestId('comparison-range-slider');
			await expect(slider).toBeVisible();
			const initialValue = await slider.inputValue();
			expect(parseInt(initialValue)).toBe(50);
			await slider.fill('75');
			await slider.dispatchEvent('input');
			await expect(slider).toHaveValue('75');
		});
	});

	test.describe('Remove/Close', () => {
		test('should remove image and return to drop zone', async ({ page, waitForWasm, uploadAndWaitForCompression }) => {
			await page.goto('/');
			await waitForWasm();
			await uploadAndWaitForCompression(FIXTURES.PNG);
			await page.getByTestId('close-button').click();
			await expect(page.getByTestId('main-content')).toHaveAttribute('data-view-mode', 'drop');
			await expect(page.getByTestId('drop-zone')).toBeVisible();
		});
	});
});
