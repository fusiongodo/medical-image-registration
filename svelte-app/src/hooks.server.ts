import { redirect, type Handle } from '@sveltejs/kit';

export const handle: Handle = async ({ event, resolve }) => {
	const path = event.url.pathname;
	if (path === '/regwsi' || path.startsWith('/regwsi/')) {
		throw redirect(308, path.replace(/^\/regwsi/, '/eval') + event.url.search);
	}
	if (path.startsWith('/api/regwsi/') || path === '/api/regwsi') {
		throw redirect(308, path.replace(/^\/api\/regwsi/, '/api/eval') + event.url.search);
	}
	return resolve(event);
};
