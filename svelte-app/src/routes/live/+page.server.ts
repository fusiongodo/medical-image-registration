import type { PageServerLoad } from './$types';
import { MAX_DEPTH } from '$lib/types';
import { pairCount } from '$lib/server/pairs';

export const load: PageServerLoad = () => {
	return { numPairs: pairCount(), maxDepth: MAX_DEPTH };
};
