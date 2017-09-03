//  NSGAII_main.java
//
//  Author:
//       Ke Li <keli.genius@gmail.com>
//       <http://www.cs.bham.ac.uk/~likw/>
//  
//	Reference:
//  	 K. Li, K. Deb, Q. Zhang, Q. Zhang, “Efficient Non-domination Level Update Method for Steady-State Evolutionary Multiobjective Optimization”, 
//       Technical Report, COIN Report No. 2015022, Michigan State University, December, 2015.
//
//  Copyright (c) 2016 Ke Li
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

import jmetal.core.Algorithm;
import jmetal.core.Operator;
import jmetal.core.Problem;
import jmetal.core.SolutionSet;
import jmetal.operators.crossover.CrossoverFactory;
import jmetal.operators.mutation.MutationFactory;
import jmetal.operators.selection.SelectionFactory;
import jmetal.problems.ProblemFactory;
import jmetal.problems.ZDT.*;
import jmetal.problems.DTLZ.*;
import jmetal.problems.WFG.*;
import jmetal.problems.M2M.*;
import jmetal.problems.LZ09.*;
import jmetal.problems.cec2009Competition.*;
import jmetal.util.Configuration;
import jmetal.util.JMException;

import java.io.File;
import java.io.IOException;
import java.util.HashMap;
import java.util.StringTokenizer;
import java.util.logging.FileHandler;
import java.util.logging.Logger;

public class NSGAII_main {
	public static Logger logger_; // Logger object
	public static FileHandler fileHandler_; // FileHandler object
	
	public static Problem initializeProblem(String name, int numberOfVariables, int numberOfObjectives) {
		Problem problem = null;
		
		if (name.equals("ZDT1"))
			problem = new ZDT1("Real", numberOfVariables); 
		else if (name.equals("ZDT2"))
			problem = new ZDT2("Real", numberOfVariables); 
		else if (name.equals("ZDT3"))
			problem = new ZDT3("Real", numberOfVariables); 
		else if (name.equals("ZDT4"))
			problem = new ZDT4("Real", numberOfVariables); 
		else if (name.equals("ZDT6"))
			problem = new ZDT6("Real", numberOfVariables); 
		else if (name.equals("DTLZ1"))
			problem = new DTLZ1("Real", numberOfVariables, numberOfObjectives); 
		else if (name.equals("DTLZ2"))
			problem = new DTLZ2("Real", numberOfVariables, numberOfObjectives); 
		else if (name.equals("DTLZ3"))
			problem = new DTLZ3("Real", numberOfVariables, numberOfObjectives); 
		else if (name.equals("DTLZ4"))
			problem = new DTLZ4("Real", numberOfVariables, numberOfObjectives); 
		else if (name.equals("DTLZ5"))
			problem = new DTLZ5("Real", numberOfVariables, numberOfObjectives); 
		else if (name.equals("DTLZ6"))
			problem = new DTLZ6("Real", numberOfVariables, numberOfObjectives); 
		else if (name.equals("DTLZ7"))
			problem = new DTLZ7("Real", numberOfVariables, numberOfObjectives); 
		
		return problem;
	}
	
	/**
	 * @param args
	 *            Command line arguments.
	 * @throws JMException
	 * @throws IOException
	 * @throws SecurityException
	 *             Usage: three options -
	 *             jmetal.metaheuristics.nsgaII.NSGAII_main -
	 *             jmetal.metaheuristics.nsgaII.NSGAII_main problemName -
	 *             jmetal.metaheuristics.nsgaII.NSGAII_main problemName
	 *             paretoFrontFile
	 */
	public static void main(String[] args) throws JMException,
			SecurityException, IOException, ClassNotFoundException {
		Problem problem; 		// The problem to solve
		Algorithm algorithm; 	// The algorithm to use
		Operator crossover; 	// Crossover operator
		Operator mutation; 		// Mutation operator
		Operator selection; 	// Selection operator

		HashMap<String, Double> parameters; 	// Operator parameters
		
		String problemName = "";
		int run_index 			= 0;
		int alg_index			= 0;
		int iter				= 0;
		int popsize				= 0;
		int numberOfVariables 	= 0;
		int numberOfObjectives  = 0;
		
		if (args.length == 1) {
			Object[] params = { "Real" };
			problem = (new ProblemFactory()).getProblem(args[0], params);
		} else if (args.length > 2) {
			run_index		   = Integer.parseInt(args[0]);
			iter 			   = Integer.parseInt(args[1]);
			popsize			   = Integer.parseInt(args[2]);
			numberOfVariables  = Integer.parseInt(args[3]);
			numberOfObjectives = Integer.parseInt(args[4]);
			problemName 	   = args[5];
			alg_index		   = Integer.parseInt(args[6]);
		} else { // Default problem
			problem = new ZDT1("Real");
		} // else	
		
		problem = initializeProblem(problemName, numberOfVariables, numberOfObjectives); 

		if (alg_index == 1)
			algorithm = new ssNSGAII_ENLU(problem);
		else
			algorithm = new ssNSGAII(problem);
		
		// get the current algorithm's name (added by Ke Li 20/04/2016)
		String algorithmName 	  = algorithm.getClass().getName();
		StringTokenizer tokenizer = new StringTokenizer(algorithmName, ".");
		int tokenizerLength 	  = tokenizer.countTokens();
		for (int i = 0; i < tokenizerLength; i++)
			algorithmName = tokenizer.nextToken();

		// Algorithm parameters
		int evaluations = 0;
		evaluations 	= popsize * iter;
		
		// Algorithm parameters	
		algorithm.setInputParameter("populationSize", popsize);
		algorithm.setInputParameter("maxEvaluations", evaluations);

		logger_ = Configuration.logger_;
		// Mutation and Crossover for Real codification
		parameters = new HashMap<String, Double>();
		parameters.put("probability", 1.0);
		parameters.put("distributionIndex", 20.0);
		crossover = CrossoverFactory.getCrossoverOperator("SBXCrossover", parameters);
		parameters = new HashMap<String, Double>();
		parameters.put("probability", 1.0 / problem.getNumberOfVariables());
		parameters.put("distributionIndex", 20.0);
		mutation = MutationFactory.getMutationOperator("PolynomialMutation", parameters);

		// Selection operator
		parameters = null;
		selection = SelectionFactory.getSelectionOperator("BinaryTournament2", parameters);

		// add the operators to the algorithm
		algorithm.addOperator("crossover", crossover);
		algorithm.addOperator("mutation", mutation);
		algorithm.addOperator("selection", selection);
		
		for (int i = run_index; i <= run_index; i++) {
			// Execute the Algorithm
			long initTime = System.currentTimeMillis();
			algorithm.setInputParameter("runIdx", i);
			System.out.println("The " + i + " run");
			SolutionSet population = algorithm.execute();
			long estimatedTime = System.currentTimeMillis() - initTime;

			// Result messages
			logger_.info("Total execution time: " + estimatedTime + "ms");
			
			jmetal.util.reportFinal.printPreferenceSolutionSet(population, i, problem.getName(), problem.getNumberOfObjectives(), algorithmName);
		}
		

	} // main
} // NSGAII_main
