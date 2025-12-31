import { test, expect, FIXTURES } from './fixtures';
import { readFileSync } from 'fs';

test.describe('Clipboard Paste', () => {
	test('should show paste hint in drop zone', async ({ page }) => {
		await page.goto('/');
		await expect(page.getByTestId('drop-zone')).toBeVisible();
		// Check that the paste hint is visible
		await expect(page.getByText('or paste from clipboard')).toBeVisible();
		await expect(page.getByText('to paste')).toBeVisible();
	});

	test('should add image from clipboard paste', async ({ page, waitForWasm }) => {
		await page.goto('/');
		await waitForWasm();

		// Read the test image file
		const imageBuffer = readFileSync(FIXTURES.PNG);
		const base64Image = imageBuffer.toString('base64');

		// Simulate paste event with an image file
		await page.evaluate(async (base64) => {
			// Convert base64 to Blob
			const byteCharacters = atob(base64);
			const byteNumbers = new Array(byteCharacters.length);
			for (let i = 0; i < byteCharacters.length; i++) {
				byteNumbers[i] = byteCharacters.charCodeAt(i);
			}
			const byteArray = new Uint8Array(byteNumbers);
			const blob = new Blob([byteArray], { type: 'image/png' });
			const file = new File([blob], 'pasted-image.png', { type: 'image/png' });

			// Create a DataTransfer object with the file
			const dataTransfer = new DataTransfer();
			dataTransfer.items.add(file);

			// Dispatch paste event
			const pasteEvent = new ClipboardEvent('paste', {
				bubbles: true,
				cancelable: true,
				clipboardData: dataTransfer,
			});
			document.dispatchEvent(pasteEvent);
		}, base64Image);

		// Wait for compression to complete
		await expect(page.getByTestId('download-button')).toBeVisible({
			timeout: 60000,
		});

		// Verify the image was added and processed
		await expect(page.getByTestId('single-view')).toBeVisible();
		await expect(page.getByTestId('image-name')).toContainText('pasted-image.png');
		await expect(page.getByTestId('compressed-image-overlay')).toBeVisible();
	});

	test('should not paste when typing in input field', async ({ page, waitForWasm, uploadAndWaitForCompression }) => {
		await page.goto('/');
		await waitForWasm();
		await uploadAndWaitForCompression(FIXTURES.PNG);

		// Enable resize to get an input field
		await page.getByTestId('resize-enabled-checkbox').click();

		// Focus on width input
		const widthInput = page.getByTestId('resize-width-input');
		await widthInput.click();
		await widthInput.focus();

		// Read test image
		const imageBuffer = readFileSync(FIXTURES.JPEG);
		const base64Image = imageBuffer.toString('base64');

		// Try to paste while focused on input - should not add new image
		await page.evaluate(async (base64) => {
			const byteCharacters = atob(base64);
			const byteNumbers = new Array(byteCharacters.length);
			for (let i = 0; i < byteCharacters.length; i++) {
				byteNumbers[i] = byteCharacters.charCodeAt(i);
			}
			const byteArray = new Uint8Array(byteNumbers);
			const blob = new Blob([byteArray], { type: 'image/jpeg' });
			const file = new File([blob], 'should-not-add.jpg', { type: 'image/jpeg' });

			const dataTransfer = new DataTransfer();
			dataTransfer.items.add(file);

			const pasteEvent = new ClipboardEvent('paste', {
				bubbles: true,
				cancelable: true,
				clipboardData: dataTransfer,
			});
			// Dispatch on the focused input element
			document.activeElement?.dispatchEvent(pasteEvent);
		}, base64Image);

		// Should still only have the original image (playground.png)
		await expect(page.getByTestId('image-name')).toContainText('playground.png');
		// Should not switch to list view (which would happen if multiple images)
		await expect(page.getByTestId('single-view')).toBeVisible();
	});

	test('should handle multiple pasted images', async ({ page, waitForWasm }) => {
		await page.goto('/');
		await waitForWasm();

		// Read test images
		const pngBuffer = readFileSync(FIXTURES.PNG);
		const jpegBuffer = readFileSync(FIXTURES.JPEG);
		const base64Png = pngBuffer.toString('base64');
		const base64Jpeg = jpegBuffer.toString('base64');

		// Simulate paste event with multiple image files
		await page.evaluate(
			async ({ png, jpeg }) => {
				// Helper to convert base64 to File
				const base64ToFile = (base64: string, filename: string, mimeType: string) => {
					const byteCharacters = atob(base64);
					const byteNumbers = new Array(byteCharacters.length);
					for (let i = 0; i < byteCharacters.length; i++) {
						byteNumbers[i] = byteCharacters.charCodeAt(i);
					}
					const byteArray = new Uint8Array(byteNumbers);
					const blob = new Blob([byteArray], { type: mimeType });
					return new File([blob], filename, { type: mimeType });
				};

				const file1 = base64ToFile(png, 'image1.png', 'image/png');
				const file2 = base64ToFile(jpeg, 'image2.jpg', 'image/jpeg');

				const dataTransfer = new DataTransfer();
				dataTransfer.items.add(file1);
				dataTransfer.items.add(file2);

				const pasteEvent = new ClipboardEvent('paste', {
					bubbles: true,
					cancelable: true,
					clipboardData: dataTransfer,
				});
				document.dispatchEvent(pasteEvent);
			},
			{ png: base64Png, jpeg: base64Jpeg }
		);

		// Wait for compression to complete
		await expect(page.getByTestId('download-button')).toBeVisible({
			timeout: 120000,
		});

		// Should switch to list view with multiple images
		await expect(page.getByTestId('list-view')).toBeVisible();
		await expect(page.getByTestId('file-count')).toContainText('2 files');
	});
});
