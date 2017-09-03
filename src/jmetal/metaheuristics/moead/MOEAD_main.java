/**
 * MOEAD_main.java
 *
 * This is the main function used to call the other specific algorithms.
 *
 * Author:
 * 		Ke Li <k.li@exeter.ac.uk>
 *
 * Affliation:
 * 		Department of Computer Science, University of Exeter
 *
 * Homepage:
 * 		https://coda-group.github.io/
 *
 * Copyright (c) 2017 Ke Li
 *
 * Note: This is a free software developed based on the open source project
 * jMetal<http://jmetal.sourceforge.net>. The copy right of jMetal belongs to
 * its original authors, Antonio J. Nebro and Juan J. Durillo. Nevertheless,
 * this current version can be redistributed and/or modified under the terms of
 * the GNU Lesser General Public License as published by the Free Software
 * Foundation, either version 3 of the License, or (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 *
 */

package jmetal.metaheuristics.moead;

import jmetal.core.*;
import jmetal.operators.crossover.*;
import jmetal.operators.mutation.*;
import jmetal.operators.selection.*;
import jmetal.problems.*;
import jmetal.problems.WFG.*;
import jmetal.problems.DTLZ.*;
import jmetal.problems.M2M.*;
import jmetal.problems.ZDT.*;
import jmetal.problems.cec2009Competition.*;
import jmetal.qualityIndicator.QualityIndicator;
import jmetal.util.Configuration;
import jmetal.util.JMException;

import java.io.File;
import java.io.IOException;
import java.util.HashMap;
import java.util.logging.FileHandler;
import java.util.logging.Logger;

public class MOEAD_main {
	public static Logger logger_; // Logger object
	public static FileHandler fileHandler_; // FileHandler object

	private static void createFolder(String str) {
		boolean success = (new File(str)).mkdir();

		if (success) {
			System.out.println("Directory: " + str + " created");
		} else {
			System.out.println("Directory: " + str + " NOT created");
		}
	}

	public static boolean deleteFolder(File dir) {
		if (dir.isDirectory()) {
			String[] children = dir.list();
			for (int i = 0; i < children.length; i++) {
				boolean success = deleteFolder(new File(dir, children[i]));
				if (!success) {
					return false;
				}
			}
		}

		// The directory is now empty so delete it
		return dir.delete();
	}

	/**
	 * @param args
	 *            Command line arguments. The first (optional) argument
	 *            specifies the problem to solve.
	 * @throws JMException
	 * @throws IOException
	 * @throws SecurityException
	 *             Usage: three options - jmetal.metaheuristics.moead.MOEAD_main
	 *             - jmetal.metaheuristics.moead.MOEAD_main problemName -
	 *             jmetal.metaheuristics.moead.MOEAD_main problemName
	 *             ParetoFrontFile
	 * @throws ClassNotFoundException
	 */
	public static void main(String[] args) throws JMException,
			SecurityException, IOException, ClassNotFoundException {
		Problem problem; 	 	// The problem to solve
		Algorithm algorithm; 	// The algorithm to use
		Operator crossover; 	// Crossover operator
		Operator mutation; 		// Mutation operator

		HashMap parameters; // Operator parameters

		QualityIndicator indicators; // Object to get quality indicators

		// Logger object and file to store log messages
		logger_ = Configuration.logger_;
		fileHandler_ = new FileHandler("MOEAD.log");
		logger_.addHandler(fileHandler_);

		indicators = null;
		if (args.length == 1) {
			Object[] params = { "Real" };
			problem 		= (new ProblemFactory()).getProblem(args[0], params);
		} // if
		else if (args.length == 2) {
			Object[] params = {"Real"};
			problem    = (new ProblemFactory()).getProblem(args[0], params);
			indicators = new QualityIndicator(problem, args[1]);
		} // if
		else { // Default problem
//			problem = new WFG9("Real");
//			problem = new UF1("Real");
//			problem = new MOP7("Real");
			problem = new DTLZ3("Real");
//			problem = new ZDT1("Real");
		} // else

//		algorithm = new MOEAD(problem);
//		algorithm = new MOEAD_DRA(problem);
//		algorithm = new MOEADDRA_MAB(problem);
		algorithm = new MOEAD_STM(problem);
//		algorithm = new MOEAD_IR(problem);
//		algorithm = new MOEADD(problem);
//		algorithm = new DPP_Pareto(problem);
//		algorithm = new DPPDRA_Pareto(problem);
//		algorithm = new DPPDRA_Epsilon(problem);
//		algorithm = new DPP_Grid(problem);
//		algorithm = new MOEAD_DRA_ASTM(problem);

		// Algorithm parameters
		algorithm.setInputParameter("populationSize", 91);
		algorithm.setInputParameter("maxEvaluations", 91 * 1000);

		algorithm.setInputParameter("dataDirectory", "weight/");
//		algorithm.setInputParameter("dataDirectory", "weight/preference");

		// Crossover operator
		int crossover_id = 2;
		if (crossover_id == 1) {
			parameters = new HashMap();
			parameters.put("CR", 0.5);
			parameters.put("F", 0.5);
			crossover = CrossoverFactory.getCrossoverOperator("DifferentialEvolutionCrossover", parameters);
		} else {
			parameters = new HashMap();
		    parameters.put("probability", 1.0) ;
		    parameters.put("distributionIndex", 20.0) ;
		    crossover = CrossoverFactory.getCrossoverOperator("SBXCrossover", parameters);
		}

		// Mutation operator
		parameters = new HashMap();
		parameters.put("probability", 1.0 / problem.getNumberOfVariables());
		parameters.put("distributionIndex", 20.0);
		mutation = MutationFactory.getMutationOperator("PolynomialMutation", parameters);

		algorithm.addOperator("crossover", crossover);
		algorithm.addOperator("mutation", mutation);

		String curDir = System.getProperty("user.dir");
		String str 	  = curDir + "/" + problem.getName() + "M" + problem.getNumberOfObjectives();

		File dir = new File(str);
		if (deleteFolder(dir)) {
			System.out.println("Folders are deleted!");
		} else
			System.out.println("Folders can NOT be deleted!");
		createFolder(str);

		String str1 = "FUN";
		String str2;
		String str3 = "VAR";
		String str4;
		for (int i = 0; i < 1; i++) {
			str2 = str1 + Integer.toString(i);
			str4 = str3 + Integer.toString(i);
			// Execute the Algorithm
			long initTime = System.currentTimeMillis();
			System.out.println("The " + i + " run");
			SolutionSet population = algorithm.execute();
			long estimatedTime = System.currentTimeMillis() - initTime;

			// Result messages
			logger_.info("Total execution time: " + estimatedTime + "ms");
			logger_.info("Variables values have been writen to file VAR");
//			population.printVariablesToFile(str4);
			logger_.info("Objectives values have been writen to file FUN");
			population.printObjectivesToFile(curDir + "/" + problem.getName()
					+ "M" + problem.getNumberOfObjectives() + "/" + str2);

			if (indicators != null) {
				logger_.info("Quality indicators");
				logger_.info("Hypervolume: "
						+ indicators.getHypervolume(population));
				logger_.info("EPSILON    : "
						+ indicators.getEpsilon(population));
				logger_.info("GD         : " + indicators.getGD(population));
				logger_.info("IGD        : " + indicators.getIGD(population));
				logger_.info("Spread     : " + indicators.getSpread(population));
			}
		}
	} // main
} // MOEAD_main
