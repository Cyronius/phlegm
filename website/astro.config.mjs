// @ts-check
import { defineConfig } from 'astro/config';
import starlight from '@astrojs/starlight';

// GitHub Pages project site; drop `base` if the site moves to a custom domain.
export default defineConfig({
	site: 'https://cyronius.github.io',
	base: '/phlegm',
	integrations: [
		starlight({
			title: 'phlegm',
			social: [{ icon: 'github', label: 'GitHub', href: 'https://github.com/Cyronius/phlegm' }],
		}),
	],
});
