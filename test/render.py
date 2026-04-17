import os
import asyncio
import nest_asyncio
from pyppeteer import launch

nest_asyncio.apply()

async def _html_to_image(html_path, output_path, image_format='png', viewport_size=(800, 800)):
    if image_format not in ['png', 'jpeg']:
        raise ValueError("Only 'png' and 'jpeg' formats are supported.")

    browser = await launch(headless=True, args=['--no-sandbox'])
    page = await browser.newPage()

    abs_path = os.path.abspath(html_path)
    await page.goto(f'file://{abs_path}', waitUntil='networkidle0')
    await page.setViewport({'width': viewport_size[0], 'height': viewport_size[1]})

    await page.screenshot({
        'path': output_path,
        'fullPage': True,
        'type': image_format
    })
    await browser.close()

def render_html_to_image(html_path, output_path, image_format='png', viewport_size=(800, 800)):
    """
    Render an HTML file to an image (PNG/JPEG) using headless Chromium.
    
    Parameters:
        html_path (str): Path to the HTML file
        output_path (str): Path to save the output image
        image_format (str): 'png' or 'jpeg'
        viewport_size (tuple): (width, height) of the viewport
    """
    asyncio.get_event_loop().run_until_complete(
        _html_to_image(html_path, output_path, image_format, viewport_size)
    )
