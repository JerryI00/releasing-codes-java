package jmetal.metaheuristics.moead;

import java.util.LinkedList;
import java.util.List;

/**
 * Created by mengyuawu3 on 20-Jun-16.
 */
public class StableMarriage {
    final int NOT_ENGAGED = -1;

    private int menSize, womenSize;
    private int[][] menPref, womenPref;

    public StableMarriage(int menSize, int womenSize, int[][] menPref, int[][] womenPref) {
        this.menSize = menSize;
        this.womenSize = womenSize;
        this.menPref = menPref;
        this.womenPref = womenPref;
    }

    /**
     * Return the stable matching between 'men' and 'women'
     * ('men' propose first). It is worth noting that the number of
     * women is larger than that of the men.
     */
    public void stableMatch(int[] menPartners, int[] womenPartners) {

        // Indicates the mating status
        for (int i = 0; i < menSize; i++)
            menPartners[i] = NOT_ENGAGED;
        for (int i = 0; i < womenSize; i++)
            womenPartners[i] = NOT_ENGAGED;

        // List of men that are not currently engaged.
        LinkedList<Integer> freeMen = new LinkedList<Integer>();
        // next[i] is the next woman to whom i has not yet proposed.
        int[] next = new int[menSize];
        for (int i = 0; i < menSize; i++) {
            freeMen.add(i);
            next[i] = 0;
        }

        while (!freeMen.isEmpty()) {
            int m = freeMen.remove();
            if (next[m] < womenSize) {
                int w = menPref[m][next[m]];
                next[m]++;
                if (womenPartners[w] == NOT_ENGAGED) {
                    menPartners[m] = w;
                    womenPartners[w] = m;
                } else {
                    int m1 = womenPartners[w];
                    if (womanPrefers(w, m, m1) == 1) {
                        menPartners[m] = w;
                        womenPartners[w] = m;
                        menPartners[m1] = NOT_ENGAGED;
                        freeMen.add(m1);
                    } else {
                        freeMen.add(m);
                    }
                }
            }
        }
    }

    /**
     * Return the stable matching with incomplete preference list between 'men' and 'women'
     * ('men' propose first). It is worth noting that the number of
     * women is larger than that of the men.
     */
    public int stableMatchIncompleteLists(int[] menPartners, int[] womenPartners, int[] womenPreferListLengths) {

        for (int i = 0; i < menSize; i++)
            menPartners[i] = NOT_ENGAGED;
        for (int i = 0; i < womenSize; i++)
            womenPartners[i] = NOT_ENGAGED;

        // List of men that are not currently engaged.
        List<Integer> candidateMen = new LinkedList<>();
        // next[i] is the next woman to whom i has not yet proposed.
        int[] next = new int[menSize];
        for (int i = 0; i < menSize; i++) {
            candidateMen.add(i);
            next[i] = 0;
        }

        while (!candidateMen.isEmpty()) {
            int m = candidateMen.remove(0);
            if (next[m] < womenSize) {
                int w = menPref[m][next[m]];
                next[m]++;
                if (womanContains(w, m, womenPreferListLengths[w])) {
                    if (womenPartners[w] == NOT_ENGAGED) {
                        menPartners[m] = w;
                        womenPartners[w] = m;
                    } else {
                        int m1 = womenPartners[w];
                        if (womanPrefers(w, m, m1) == 1) {
                            menPartners[m] = w;
                            womenPartners[w] = m;
                            menPartners[m1] = NOT_ENGAGED;
                            candidateMen.add(m1);
                        } else {
                            candidateMen.add(m);
                        }
                    }
                } else {
                    candidateMen.add(m);
                }
            }
        }

        int matchSize = 0;
        for (int i = 0; i < menSize; i++) {
            if (menPartners[i] != NOT_ENGAGED)
                matchSize++;
        }
//        int matchSize2 = 0;
//        for (int i = 0; i < womenSize; i++) {
//            if (womenPartners[i] != NOT_ENGAGED)
//                matchSize2++;
//        }
//        if (matchSize != matchSize2) {
//            int a = 1;
//        }
        return matchSize;
    }

    public void stableMatchTwoLevel(int[] menPartners, int[] womenPartners, int[] womenPreferListLengths) {

        int matchSize = stableMatchIncompleteLists(menPartners, womenPartners, womenPreferListLengths);

        int unmatchedMenSize = menSize - matchSize;
        int unmatchedWomenSize = womenSize - matchSize;
        if (unmatchedMenSize > 0 && unmatchedWomenSize > 0) {
            int uMenPref[][] = new int[unmatchedMenSize][unmatchedWomenSize];
            int uWomenPref[][] = new int[unmatchedWomenSize][unmatchedMenSize];
            int unmatchedMen[] = new int[unmatchedMenSize];
            int unmatchedWomen[] = new int[unmatchedWomenSize];
            int unmatchedMenIndexes[] = new int[menSize];
            int unmatchedWomenIndexes[] = new int[womenSize];

            int k = 0;
            for (int i = 0; i < menSize; i++) {
                if (menPartners[i] == NOT_ENGAGED) {
                    unmatchedMen[k] = i;
                    unmatchedMenIndexes[i] = k;
                    k++;
                }
            }
            k = 0;
            for (int i = 0; i < womenSize; i++) {
                if (womenPartners[i] == NOT_ENGAGED) {
                    unmatchedWomen[k] = i;
                    unmatchedWomenIndexes[i] = k;
                    k++;
                }
            }
            for (int i = 0; i < unmatchedMenSize; i++) {
                k = 0;
                for (int j = 0; j < womenSize; j++) {
                    if (womenPartners[menPref[unmatchedMen[i]][j]] == NOT_ENGAGED) {
                        uMenPref[i][k] = unmatchedWomenIndexes[menPref[unmatchedMen[i]][j]];
                        k++;
                    }
                }
            }
            for (int i = 0; i < unmatchedWomenSize; i++) {
                k = 0;
                for (int j = 0; j < menSize; j++) {
                    if (menPartners[womenPref[unmatchedWomen[i]][j]] == NOT_ENGAGED) {
                        uWomenPref[i][k] = unmatchedMenIndexes[womenPref[unmatchedWomen[i]][j]];
                        k++;
                    }
                }
            }

            StableMarriage uSmp = new StableMarriage(unmatchedMenSize, unmatchedWomenSize, uMenPref, uWomenPref);
            int[] uMenPartners = new int[unmatchedMenSize];
            int[] uWomenPartners = new int[unmatchedWomenSize];
            uSmp.stableMatch(uMenPartners, uWomenPartners);

            for (int i = 0; i < unmatchedMenSize; i++) {
                if (uMenPartners[i] != NOT_ENGAGED) {
                    menPartners[unmatchedMen[i]] = unmatchedWomen[uMenPartners[i]];
                }
            }
            for (int i = 0; i < unmatchedWomenSize; i++) {
                if (uWomenPartners[i] != NOT_ENGAGED) {
                    womenPartners[unmatchedWomen[i]] = unmatchedMen[uWomenPartners[i]];
                }
            }
        }
    }

    public void stableMatchTwoLevel(int[] menPartners, int[] womenPartners, int[] menPartnersFirstLevel, int[] womenPartnersFirstLevel, int[] womenPreferListLengths) {

        int matchSize = stableMatchIncompleteLists(menPartners, womenPartners, womenPreferListLengths);
        for (int i = 0; i < menSize; i++) {
            menPartnersFirstLevel[i] = menPartners[i];
        }
        for (int i = 0; i < womenSize; i++) {
            womenPartnersFirstLevel[i] = womenPartners[i];
        }

        int unmatchedMenSize = menSize - matchSize;
        int unmatchedWomenSize = womenSize - matchSize;
        if (unmatchedMenSize > 0 && unmatchedWomenSize > 0) {
            int uMenPref[][] = new int[unmatchedMenSize][unmatchedWomenSize];
            int uWomenPref[][] = new int[unmatchedWomenSize][unmatchedMenSize];
            int unmatchedMen[] = new int[unmatchedMenSize];
            int unmatchedWomen[] = new int[unmatchedWomenSize];
            int unmatchedMenIndexes[] = new int[menSize];
            int unmatchedWomenIndexes[] = new int[womenSize];

            int k = 0;
            for (int i = 0; i < menSize; i++) {
                if (menPartners[i] == NOT_ENGAGED) {
                    unmatchedMen[k] = i;
                    unmatchedMenIndexes[i] = k;
                    k++;
                }
            }
            k = 0;
            for (int i = 0; i < womenSize; i++) {
                if (womenPartners[i] == NOT_ENGAGED) {
                    unmatchedWomen[k] = i;
                    unmatchedWomenIndexes[i] = k;
                    k++;
                }
            }
            for (int i = 0; i < unmatchedMenSize; i++) {
                k = 0;
                for (int j = 0; j < womenSize; j++) {
                    if (womenPartners[menPref[unmatchedMen[i]][j]] == NOT_ENGAGED) {
                        uMenPref[i][k] = unmatchedWomenIndexes[menPref[unmatchedMen[i]][j]];
                        k++;
                    }
                }
            }
            for (int i = 0; i < unmatchedWomenSize; i++) {
                k = 0;
                for (int j = 0; j < menSize; j++) {
                    if (menPartners[womenPref[unmatchedWomen[i]][j]] == NOT_ENGAGED) {
                        uWomenPref[i][k] = unmatchedMenIndexes[womenPref[unmatchedWomen[i]][j]];
                        k++;
                    }
                }
            }

            StableMarriage uSmp = new StableMarriage(unmatchedMenSize, unmatchedWomenSize, uMenPref, uWomenPref);
            int[] uMenPartners = new int[unmatchedMenSize];
            int[] uWomenPartners = new int[unmatchedWomenSize];
            uSmp.stableMatch(uMenPartners, uWomenPartners);

            for (int i = 0; i < unmatchedMenSize; i++) {
                if (uMenPartners[i] != NOT_ENGAGED) {
                    menPartners[unmatchedMen[i]] = unmatchedWomen[uMenPartners[i]];
                }
            }
            for (int i = 0; i < unmatchedWomenSize; i++) {
                if (uWomenPartners[i] != NOT_ENGAGED) {
                    womenPartners[unmatchedWomen[i]] = unmatchedMen[uWomenPartners[i]];
                }
            }
        }
    }

    /**
     * Returns true in case that a given woman prefers x to y.
     */
    private int womanPrefers(int w, int m1, int m2) {

        for (int i = 0; i < menSize; i++) {
            int pref = womenPref[w][i];
            if (pref == m1)
                return 1;
            if (pref == m2)
                return -1;
        }
        // This should never happen.
        System.out.println("Error in womanPref list!");
        return 0;
    }

    /**
     * Check if woman w contains m on her preference list
     */
    private boolean womanContains(int w, int m, int preferListLength) {
        for (int i = 0; i < preferListLength; i++) {
            if (womenPref[w][i] == m)
                return true;
        }
        return false;
    }

    public int getMenSize() {
        return menSize;
    }

    public int getWomenSize() {
        return womenSize;
    }
}
