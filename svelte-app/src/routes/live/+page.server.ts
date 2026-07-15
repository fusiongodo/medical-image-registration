import type { PageServerLoad } from './$types';
import { NUM_PAIRS, MAX_DEPTH } from '$lib/types';

export const load: PageServerLoad = () => {
	return { numPairs: NUM_PAIRS, maxDepth: MAX_DEPTH };
};
