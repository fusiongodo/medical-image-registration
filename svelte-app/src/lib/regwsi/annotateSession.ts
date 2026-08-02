export type AnnotateSide = 'he' | 'ihc';

export type Landmark = {
	he: [number, number];
	ihc: [number, number];
};

export type AnnotateSession = {
	phase: AnnotateSide;
	pendingHe: [number, number] | null;
	landmarks: Landmark[];
	rev: number;
};

export function emptySession(landmarks: Landmark[] = []): AnnotateSession {
	return { phase: 'he', pendingHe: null, landmarks, rev: 0 };
}

export function storageKey(pairId: number) {
	return `regwsi-annotate-${pairId}`;
}

export function channelName(pairId: number) {
	return `regwsi-annotate-${pairId}`;
}

export function readStoredSession(pairId: number): AnnotateSession | null {
	if (typeof localStorage === 'undefined') return null;
	try {
		const raw = localStorage.getItem(storageKey(pairId));
		if (!raw) return null;
		const data = JSON.parse(raw) as AnnotateSession;
		if (data.phase !== 'he' && data.phase !== 'ihc') return null;
		if (!Array.isArray(data.landmarks)) return null;
		return {
			phase: data.phase,
			pendingHe: data.pendingHe ?? null,
			landmarks: data.landmarks,
			rev: typeof data.rev === 'number' ? data.rev : 0
		};
	} catch {
		return null;
	}
}

export function writeStoredSession(pairId: number, session: AnnotateSession) {
	if (typeof localStorage === 'undefined') return;
	localStorage.setItem(storageKey(pairId), JSON.stringify(session));
}

export type SessionHandle = {
	publish: (
		current: AnnotateSession,
		update: Partial<Omit<AnnotateSession, 'rev'>>
	) => AnnotateSession;
	close: () => void;
};

export function connectSession(
	pairId: number,
	onRemote: (session: AnnotateSession) => void
): SessionHandle {
	const ch =
		typeof BroadcastChannel !== 'undefined' ? new BroadcastChannel(channelName(pairId)) : null;

	const onMessage = (ev: MessageEvent<AnnotateSession>) => {
		if (!ev.data || typeof ev.data.rev !== 'number') return;
		onRemote(ev.data);
	};
	ch?.addEventListener('message', onMessage);

	const onStorage = (ev: StorageEvent) => {
		if (ev.key !== storageKey(pairId) || !ev.newValue) return;
		try {
			const data = JSON.parse(ev.newValue) as AnnotateSession;
			if (typeof data.rev === 'number') onRemote(data);
		} catch {
			/* ignore */
		}
	};
	if (typeof window !== 'undefined') window.addEventListener('storage', onStorage);

	return {
		publish(current, update) {
			const session: AnnotateSession = {
				phase: update.phase ?? current.phase,
				pendingHe: update.pendingHe !== undefined ? update.pendingHe : current.pendingHe,
				landmarks: update.landmarks ?? current.landmarks,
				rev: current.rev + 1
			};
			writeStoredSession(pairId, session);
			ch?.postMessage(session);
			return session;
		},
		close() {
			ch?.removeEventListener('message', onMessage);
			ch?.close();
			if (typeof window !== 'undefined') window.removeEventListener('storage', onStorage);
		}
	};
}
