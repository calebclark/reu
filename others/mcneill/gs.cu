static const uint N = 10;

__device__ int high_bit_pos(uint x) {
	return 32 - __clz(x);
}

__global__
void parallel_gale_shapely(uint8_t **female_prefences,
	                       uint8_t **female_pref_list,
		                   uint8_t **male_pref_list,
		                   uint8_t *result) {

	/* TODO copy female_preferences and male_pref_list into shared */

	__shared__ uint16_t proposals[N];
	proposals[threadIdx.x] = 0;

	__shared__ bool male_engaged[N];
	male_engaged[threadIdx.x] = false;
	uint16_t female_engaged = 0;

	uint8_t nextPref = 0;

	__syncthreads();
	for (int i=0; i<N; i++) {

		/* each male determines who to propose to (if he's not 
		 * already engaged) */
		if (!male_engaged[threadIdx.x]) {
			uint8_t proposee = male_pref_list[threadIdx.x][nextPref];
			nextPref++;

			atomicAdd(engagements+proposee, 
					  1 << female_preferences[proposee][threadIdx.x]);
		}

		__syncthreads();

		/* each female chooses her engagement */
		if (engagements[threadIdx.x] > female_engaged) {
			/* she calls off her old engagement... */
			male_engaged[female_engaged] = false;

			/* ...and is now engaged to her favoirite new suiter */
			female_engaged = female_pref_list[
				                high_bit_pos(engagements[threadIdx.x])];
			male_engaged[female_engaged] = true;
		}

		__syncthreads();
	}
}
