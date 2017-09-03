package jmetal.metaheuristics.moead;

import jmetal.core.SolutionSet;

public class QuickSort {
	/**
	 * Sort the current "SlidingWindow" based on the achieved fitness values
	 * 
	 * @param SlidingWindow
	 * @param curSize
	 */
	public static void quickSlideWindow(double[][] SlidingWindow, int curSize) {
		sortSlideWindow(SlidingWindow, 0, SlidingWindow[0].length - 1);
	}

	private static void sortSlideWindow(double[][] SlidingWindow, int left,
			int right) {
		if (left < right) {
			int i = left;
			int j = right + 1;
			while (true) {
				// Search left
				while (i + 1 < SlidingWindow[0].length
						&& SlidingWindow[1][++i] > SlidingWindow[1][left])
					;
				while (j - 1 > -1
						&& SlidingWindow[1][--j] < SlidingWindow[1][left])
					;
				if (i >= j)
					break;
				swapSlideWindow(SlidingWindow, i, j);
			}
			swapSlideWindow(SlidingWindow, left, j);
			sortSlideWindow(SlidingWindow, left, j - 1); // Cycle in the left side
			sortSlideWindow(SlidingWindow, j + 1, right); // Cycle in the right side
		}
	}

	private static void swapSlideWindow(double[][] SlidingWindow, int i, int j) {
		double temp_index, temp_value;

		temp_index = SlidingWindow[0][i];
		SlidingWindow[0][i] = SlidingWindow[0][j];
		SlidingWindow[0][j] = temp_index;

		temp_value = SlidingWindow[1][i];
		SlidingWindow[1][i] = SlidingWindow[1][j];
		SlidingWindow[1][j] = temp_value;

		return;
	}

	private static void swapDist(int[] dist, int i, int j) {
		int t = dist[i];
		dist[i] = dist[j];
		dist[j] = t;
	}

	public static void quickObj(SolutionSet population_, int[] dist, int size) {
		sortObj(population_, dist, 0, size - 1);
	}

	private static void sortObj(SolutionSet population_, int[] dist, int left,
			int right) {
		if (left < right) {
			int i = left;
			int j = right + 1;
			while (true) {
				// Search left
				while (i + 1 < population_.size()
						&& population_.get(dist[++i]).getObjective(0) > population_
								.get(dist[left]).getObjective(0))
					;
				while (j - 1 > -1
						&& population_.get(dist[--j]).getObjective(0) > population_
								.get(dist[left]).getObjective(0))
					;
				if (i >= j)
					break;
				swapDist(dist, i, j);
			}
			swapDist(dist, left, j);
			sortObj(population_, dist, left, j - 1); // Cycle in the left side
			sortObj(population_, dist, j + 1, right); // Cycle in the right side
		}
	}
}