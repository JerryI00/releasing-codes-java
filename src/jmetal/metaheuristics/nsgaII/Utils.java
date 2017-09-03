//  Utils.java
//
//  Author:
//       Antonio J. Nebro <antonio@lcc.uma.es>
//       Juan J. Durillo <durillo@lcc.uma.es>
//
//  Copyright (c) 2011 Antonio J. Nebro, Juan J. Durillo
//
//  This program is free software: you can redistribute it and/or modify
//  it under the terms of the GNU Lesser General Public License as published by
//  the Free Software Foundation, either version 3 of the License, or
//  (at your option) any later version.
//
//  This program is distributed in the hope that it will be useful,
//  but WITHOUT ANY WARRANTY; without even the implied warranty of
//  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
//  GNU Lesser General Public License for more details.
// 
//  You should have received a copy of the GNU Lesser General Public License
//  along with this program.  If not, see <http://www.gnu.org/licenses/>.

package jmetal.metaheuristics.nsgaII;

/**
 * Utilities methods to used by MOEA/D
 */
public class Utils {

	public static int[] QuickSort(double[] pData, int[] idxArray, int left, int right) {
		int i, j;
		int midIdx, tempIdx;
		double middle, temp;
		
		i = left;
		j = right;

		middle = pData[left];
		midIdx = idxArray[left];
		while (true) {
			while ((++i) < right - 1 && pData[i] < middle)
				;
			while ((--j) > left && pData[j] > middle)
				;
			if (i >= j)
				break;
			temp = pData[i];
			pData[i] = pData[j];
			pData[j] = temp;

			tempIdx = idxArray[i];
			idxArray[i] = idxArray[j];
			idxArray[j] = tempIdx;
		}

		pData[left] = pData[j];
		pData[j] 	= middle;

		idxArray[left] = idxArray[j];
		idxArray[j]    = midIdx;

		if (left < j)
			QuickSort(pData, idxArray, left, j);

		if (right > i)
			QuickSort(pData, idxArray, i, right);
		
		return idxArray;
	}
	
	public static int exist(int[] idxArray, int curIdx) {
		int flag = 0;
		
		for (int i = 0; i < idxArray.length; i++) {
			if (idxArray[i] == curIdx) {
				flag = 1;
				break;
			}
		}
		
		return flag;
	}

	public static double distVector(double[] vector1, double[] vector2) {
		int dim = vector1.length;
		
		double sum = 0;
		for (int n = 0; n < dim; n++)
			sum += (vector1[n] - vector2[n]) * (vector1[n] - vector2[n]);

		return Math.sqrt(sum);
	} // distVector

	public static void minFastSort(double x[], int idx[], int n, int m) {
		for (int i = 0; i < m; i++) {
			for (int j = i + 1; j < n; j++) {
				if (x[i] > x[j]) {
					double temp = x[i];
					x[i] = x[j];
					x[j] = temp;
					int id = idx[i];
					idx[i] = idx[j];
					idx[j] = id;
				} // if
			}
		} // for

	} // minFastSort

	public static void randomPermutation(int[] perm, int size) {
		int[] index = new int[size];
		boolean[] flag = new boolean[size];

		for (int n = 0; n < size; n++) {
			index[n] = n;
			flag[n] = true;
		}

		int num = 0;
		while (num < size) {
			int start = jmetal.util.PseudoRandom.randInt(0, size - 1);
			// int start = int(size*nd_uni(&rnd_uni_init));
			while (true) {
				if (flag[start]) {
					perm[num] = index[start];
					flag[start] = false;
					num++;
					break;
				}
				if (start == (size - 1)) {
					start = 0;
				} else {
					start++;
				}
			}
		} // while
	} // randomPermutation

	/**
	 * Calculate the dot product of two vectors
	 * 
	 * @param vec1
	 * @param vec2
	 * @return
	 */
	public static double innerproduct(double[] vec1, double[] vec2) {
		double sum = 0;

		for (int i = 0; i < vec1.length; i++)
			sum += vec1[i] * vec2[i];

		return sum;
	}

	/**
	 * Calculate the norm of the vector
	 * 
	 * @param z
	 * @return
	 */
	public static double norm_vector(double[] z, int numberObjectives) {
		double sum = 0;

		for (int i = 0; i < numberObjectives; i++)
			sum += z[i] * z[i];

		return Math.sqrt(sum);
	}
}
