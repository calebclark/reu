import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.FileReader;
import java.io.PrintStream;
import java.util.ArrayList;
import java.util.List;
import java.util.Scanner;

/*

The data prints out to "data.txt".
There are also some messages that pop up in the console, but they aren't important.

You can see how far the program is by looking at data.txt, which prints a line every time it does 1 million trials.


If you finish some n values and need to start the program again, just change nMin and nMax below

 */

public class ImprovedPII {

	public static void main(String[] args) throws FileNotFoundException {


		int trials = 1000;
		int nMin = 10; 
		int nInc = 10;
		int nMax = 100; 
		
		
		int c = 6;
		boolean print = false; //prints various matrices
		String create = "true"; //true,sequential,hadoop,exhaustive
		boolean smartinit = false; //smart initiation improvement
		boolean init2 = false; //cuts off first phase after n iterations, starts transposed phase 
		boolean cycDetect = false; //cycle detection code

		PrintStream ps = new PrintStream(new FileOutputStream("data.txt"));
		int countTotal = 0; //total count of all trials
		int togTotal = 0; //count of successful trials
		
		int test = 2; //for smartinit 1 and 2 (just leave this at 2)
		int[] totalSteps = new int[test];
		int[] totals = new int[test];

		for(int n=nMin;n<=nMax;n=n+nInc) {
			int count = 0; //count per n value
			int longest = 0; //longest cycle
			int mostSteps[] = {0,0};
			int[] counts = new int[test];
			int[] steps = new int[test];

			for(int t=0;t<trials;t++) {
				if(t%1000000==0 && t != 0)
					ps.println("million "+(counts[0]+counts[1]));
				int[][][] initGraph;
				int[] initiMatch = new int[n];
				int[] initjMatch = new int[n];

				if(create.equals("true")) {  //create random preference list and initial match
					initGraph = create(n); //create random preference list

					//initiation step    graph[i][j][l,r,type,pointer]
					for(int i=0;i<n-1;i++) {  // (i,i) trades pointers with a random (i,j) i<=j<n
						int j = i+(int)(Math.random()*(n-i));  // (i,i) points to (i+1,j)   (i,j) points to (i+1,i)
						initGraph[i][i][3] = j;
						initGraph[i][j][3] = i;
					}
					for(int i=0;i<n;i++) { //find the match for each row
						int j = initGraph[0][i][3];
						for(int k=1;k<n;k++) { //go through the chain of pointers
							j = initGraph[k][j][3];
						}
						initGraph[i][j][2] = 1; //1 = match
						initiMatch[i] = j;
						initjMatch[j] = i;
					} //initiation done
				}
				else if(create.equals("sequential")){ // inputs an existing preference list and initial match from init.txt
					initGraph = getInit(n); //gets the preference list and initial match
					for(int i=0;i<n;i++) { //sets up the arrays of matches
						for(int j=0;j<n;j++) {
							if(initGraph[i][j][2] == 1) {
								initiMatch[i] = j;
								initjMatch[j] = i;
							}
						}
					}
				}
				else if(create.equals("hadoop")){ //inputs an existing preference list and initial match from part-00000 (file from hadoop)
					initGraph = getHadoop(n); //gets the preference list and initial match
					for(int i=0;i<n;i++) { //sets up the arrays of matches
						for(int j=0;j<n;j++) {
							if(initGraph[i][j][2] == 1) {
								initiMatch[i] = j;
								initjMatch[j] = i;
							}
						}
					}
				}
				else {
					System.out.println("misspelled");
					return;
				}  //done creating preferences and initial matching

				if(print)
					print(initGraph,n,"init.txt");

				int[][][] graph = new int[n][n][4]; //the actual graph we will be using
				int[] iMatch = new int[n]; // (i,iMatch[i]) is a match
				int[] jMatch = new int[n]; // (jMatch[j],j) is a match
				int stepNum;

				boolean cycled = false;
				boolean matchFromCycle = false;
				boolean past3 = false;
				boolean[][] repeats = new boolean[n][n]; //matrix that makes sure nm1s are not repeated in chains

				for(int a=0;a<2;a++) {  ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
					//this loop exists because running smartinit twice, once on l and once on r values, finds a stable match with very high probability
					//I was going to run the two trials in tandem, but never got around to it
					stepNum = 0;

					if(a==0 && !init2) //if we're not running smartinit twice, just do this loop once
						a=1;

					for(int i=0;i<n;i++) { //go back to initial graph
						for(int j=0;j<n;j++) {
							for(int k=0;k<4;k++) {
								graph[i][j][k] = initGraph[i][j][k];
							}
						}
						iMatch[i] = initiMatch[i];
						jMatch[i] = initjMatch[i];
					}


					if(smartinit) { //smart initiation overrides the current initial match
						Object[] newMatches = null;
						if(a==0)
							newMatches = smartInit(graph,n,1); //runs on r values
						else
							newMatches = smartInit(graph,n,0); //runs on l values

						iMatch = (int[])newMatches[0];
						jMatch = (int[])newMatches[1];
						if(print) { //print the initial matching
							for(int i=0;i<n;i++) {
								System.out.println("i:"+i+" "+iMatch[i]);
							}
						}
					}

					boolean stable = true;
					int[] firstMatch = new int[n];  //to see the initial match in case of errors
					int[][] cycle = new int[3*n+1][n]; //to detect cycles
					List<List<int[]>> nm1s = new ArrayList<List<int[]>>(); //nm1 chains
					List<Integer> parity = new ArrayList<Integer>(); //nm1 chain parities

					boolean[] roots = new boolean[n]; //to check for roots in cycles
					for(int i=0;i<n;i++) { //start them all as true
						roots[i] = true;
					}

					iter:
					for(int s=0;s<c*n;s++) { ////////////////////////////////////////////////////////////////////       iteration loop       /////////////////////////

						  /*
						  if(s==0) {     //used for debugging specific initial matches
							  iMatch[0] = 5;
							  iMatch[1] = 2;
							  iMatch[2] = 7;
							  iMatch[3] = 0;
							  iMatch[4] = 3;
							  iMatch[5] = 1;
							  iMatch[6] = 4;
							  iMatch[7] = 6;

							  jMatch[5] = 0;
							  jMatch[2] = 1;
							  jMatch[7] = 2;
							  jMatch[0] = 3;
							  jMatch[3] = 4;
							  jMatch[1] = 5;
							  jMatch[4] = 6;
							  jMatch[6] = 7;

						  }

						  for(int i=0;i<n;i++) {
							  System.out.print(iMatch[i]+" ");
						  }
						  System.out.println(" "+s);
						  */

						  if(s==3*n)
							  past3 = true;
						  if(past3 && s < 6*n+1 && !cycled && cycDetect) {   //if s>=3*n, start looking for cycles. the matrix only has 3*n slots
							  for(int i=0;i<n;i++) { //update cycle
								  cycle[s-3*n][i] = iMatch[i];
							  }

							  boolean same = true;
							  if(s==3*n) //don't check for cycles on the first iteration
								  same = false;
							  for(int i=0;i<n;i++) { //check for cycles
								  if(cycle[s-3*n][i] != cycle[0][i]) {
									  same = false;
									  roots[i] = false; //check for pairs that don't change
								  }
							  }

							  if(same) { //if there's a cycle
								  cycled = true;
								  if(s-3*n > longest) //keep track of longest cycle
									  longest = s-3*n;
							  }
						  } //out of past3

						  if(cycled && !matchFromCycle && cycDetect) { //if a cycle was detected, and haven't been here before
							  matchFromCycle = true;

							  for(int i=0;i<n;i++) { //reset the matches
								  iMatch[i] = -1;
								  jMatch[i] = -1;
							  }

							  for(int i=0;i<nm1s.size();i++) { //check if we need to merge the lists together
								  List<int[]> nm1 = nm1s.get(i);
								  int iValue = nm1.get(0)[0];
								  int jValue = nm1.get(0)[1];
								  for(int j=0;j<nm1s.size();j++) {
									  List<int[]> nm1Check = nm1s.get(j);
									  if(i != j && (iValue == nm1Check.get(nm1Check.size()-1)[0] || jValue == nm1Check.get(nm1Check.size()-1)[1])) { //if we do need to merge
										  if(nm1Check.size() > 1 && (nm1Check.get(nm1Check.size()-2)[0] == iValue && nm1Check.get(nm1Check.size()-1)[0] == iValue ||
												  nm1Check.get(nm1Check.size()-2)[1] == jValue && nm1Check.get(nm1Check.size()-1)[1] == jValue)) { //if we need to change parity 2-1
											  parity.remove(j);
											  parity.add(j,(nm1Check.size()-1)%2);
											  //System.out.println("switching parity in merge 2-1 for i "+iValue+" nm1i "+jValue);
										  }
										  else if(nm1.size() > 1 && (nm1Check.get(nm1Check.size()-1)[0] == iValue && nm1.get(1)[0] == iValue ||
												  nm1Check.get(nm1Check.size()-1)[1] == jValue && nm1.get(1)[1] == jValue)) { //if we need to change parity 1-2 case
											  parity.remove(j);
											  parity.add(j,nm1Check.size()%2);
											  //System.out.println("switching parity in merge 1-2 for i "+iValue+" nm1i "+jValue);
										  }
										  if(parity.get(i) != -1) { //update the parity of the merged list
											  //System.out.println("updating parity in merge");
											  int par = (parity.get(i) + nm1Check.size())%2;
											  parity.remove(j);
											  parity.add(j,par);
										  }
										  //System.out.println("merging at end"+i+" "+j);
										  while(nm1.size() != 0) {  //put nm1check2 on the end of nm1check
											  nm1Check.add(nm1.remove(0));
										  }
										  nm1s.remove(i);
										  parity.remove(i);
										  i--; //now we have to go back a chain
										  break;
									  }
								  }
							  }

							  if(nm1s.size()>1) { //check if we need to merge dead ends
								  for(int i=0;i<nm1s.size();i++) { //for all chains
									  List<int[]> nm1 = nm1s.get(i);
									  int iValue = nm1.get(0)[0];
									  int jValue = nm1.get(0)[1];
									  int adjacent = 0; 
									  for(int k=0;k<n;k++) {
										  if(repeats[iValue][k] && k != jValue)
											  adjacent++;
									  }
									  for(int k=0;k<n;k++) {
										  if(repeats[k][jValue] && k != iValue)
											  adjacent++;
									  }
									  if(adjacent < 2) { //if it's a dead end
										  iValue = nm1.get(nm1.size()-1)[0];
										  jValue = nm1.get(nm1.size()-1)[1];
										  for(int j=0;j<nm1s.size();j++) { //check every other chain
											  List<int[]> nm1Check = nm1s.get(j);
											  if(j != i && (nm1Check.get(0)[0] == nm1Check.get(nm1Check.size()-1)[0] ||
													  nm1Check.get(0)[1] == nm1Check.get(nm1Check.size()-1)[1])) { //if the chain is a cycle
												  for(int k=0;k<nm1Check.size();k++) { //check every nm1 for an opening
													  if(iValue == nm1Check.get(k)[0] || jValue == nm1Check.get(k)[1]) { //if there is an opening
														  //System.out.println("merging dead end");
														  while(k+1<nm1Check.size()) { //add everything after k
															  nm1.add(nm1Check.remove(k+1));
														  }
														  while(nm1Check.size() > 0) { //add everything else
															  nm1.add(nm1Check.remove(0));
														  }
														  parity.remove(i);
														  parity.add(i,0);
														  nm1s.remove(j);
														  parity.remove(j);
														  break;
													  }
												  }
											  }
										  }
									  }
								  }
							  } //done with dead end merge

							  if(nm1s.size() > 2) { //either something is wrong, or there are three cycles
								  //print everything to a file and break to next trial
								  //System.out.println("found "+nm1s.size()+" nm1s, outputting to "+nm1s.size()+"nm1s.txt");
								  
								  //sometimes it is useful to generate a file with this odd error
								  /*
								  PrintStream nm = new PrintStream(new FileOutputStream(nm1s.size()+"nm1s.txt"));

								  for(int i=0;i<n;i++) {  //print graph
									  for(int j=0;j<n;j++) {
										  nm.print(graph[i][j][0]+" "+graph[i][j][1]);
										  nm.print("\t");
									  }
									  nm.println();
								  }
								  nm.println();
								  nm.println();
								  nm.println("cycles");
								  for(int k=0;k<2*n-1;k++) {  //print cycles
									  for(int j=0;j<n;j++) {
										  nm.print(cycle[k][j]+" ");
									  }
									  nm.println();
								  }
								  nm.print("first Match:");
								  for(int i=0;i<n;i++) {
									  nm.print(firstMatch[i]+" ");
								  }
								  nm.println();
								  nm.print("iMatch:");
								  for(int i=0;i<n;i++) {
									  nm.print(iMatch[i]);
								  }
								  nm.println();
								  nm.println("printing nm1s");
								  for(int i=0;i<nm1s.size();i++) {
									  List<int[]> debug = nm1s.get(i);
									  for(int j=0;j<debug.size();j++) {
										  nm.print(debug.get(j)[0]+" "+debug.get(j)[1]+",");
									  }
									  nm.println();
								  }
								  */
								  
								  stable = false;
								  break iter;
							  }

							  for(int i=0;i<nm1s.size();i++) { //add every other pair in the loops to iMatch
								  List<int[]> nm1Check = nm1s.get(i);
								  int offset = parity.get(i); //this is when there are three nm1s in the same row/column
								  if(offset == -1) //if it wasn't set
									  offset = 0;
								  int j=0;
								  while(2*j+offset<nm1Check.size()) {
									  iMatch[nm1Check.get(2*j+offset)[0]] = nm1Check.get(2*j+offset)[1]; //set iMatch
									  j++;
								  }
							  }

							  for(int i=0;i<n;i++) { //add "roots" to iMatch (pairs that stay the same are not in the nm1 loops)
								  if(roots[i]) { //if it's a root
									  if(iMatch[i] == -1) // if there's no nm1 in the way
										  iMatch[i] = cycle[0][i]; //set iMatch
									  else {
										  //System.out.println("root and nm1 in same place");
										  return;
									  }
								  }
							  }

							  int badCount = 0; //check if any row doesn't have a match
							  for(int i = 0;i<n;i++) {
								  if(iMatch[i] == -1) {
									  badCount++;
								  }
								  else
									  jMatch[iMatch[i]] = i; //set up jMatch
							  }
							  if(badCount == 1) { //fill in the gap
								  int lasti = -1;
								  int lastj = -1;
								  for(int i=0;i<n;i++) { //find the gap
									  if(iMatch[i] == -1)
										  lasti = i;
									  if(jMatch[i] == -1)
										  lastj = i;
								  }
								  iMatch[lasti] = lastj; //fill it in
								  jMatch[lastj] = lasti;

							  }
							  else if(badCount != 0) {    //if >1 gap, print to file and break
								  
								  //again, sometimes we want to print this out because it is an unusual error
								  
								  /*
								  //System.out.println("badCount:"+badCount+" outputting to "+"badCount"+n+" "+badCount+".txt");
								  PrintStream bc = new PrintStream(new FileOutputStream("badCount"+n+" "+badCount+".txt"));

								  for(int i=0;i<n;i++) { //print graph
									  for(int j=0;j<n;j++) {
										  bc.print(graph[i][j][0]+" "+graph[i][j][1]);
										  bc.print("\t");
									  }
									  bc.println();
								  }
								  bc.println();
								  bc.println();
								  bc.println("cycles");
								  for(int k=0;k<2*n-1;k++) {  //print cycles
									  for(int j=0;j<n;j++) {
										  bc.print(cycle[k][j]+" ");
									  }
									  bc.println();
								  }
								  bc.print("first Matching:");
								  for(int i=0;i<n;i++) { //print the first matching
									  bc.print(firstMatch[i]+" ");
								  }
								  bc.println();
								  bc.print("iMatch:");
								  for(int i=0;i<n;i++) { //print the last matching
									  bc.print(iMatch[i]);
								  }
								  */
								  stable = false;
								  break iter;
							  }
						  } //done with cycle detection code

						  boolean okay = true; //test jMatch before entering the iteration loop
						  for(int i=0;i<n;i++) {
							  for(int j=0;j<n;j++) {
								  if(iMatch[i] == -1 || jMatch[i] == -1) {
									  okay = false;
								  }
							  }
						  }
						  if(!okay) { //if there was a problem in jMatch

							  //System.out.println("matches are not set up, printing to notset");
							  
							  
							  //yet another unusual error that we might want to print out
							  
							  /*
							  PrintStream ns = new PrintStream(new FileOutputStream("notset.txt"));

							  for(int i=0;i<n;i++) { //print graph
								  for(int j=0;j<n;j++) {
									  ns.print(graph[i][j][0]+" "+graph[i][j][1]);
									  ns.print("\t");
								  }
								  ns.println();
							  }
							  ns.println();
							  ns.println();
							  ns.println("cycles");
							  for(int k=0;k<2*n-1;k++) {  //print cycles
								  for(int j=0;j<n;j++) {
									  ns.print(cycle[k][j]+" ");
								  }
								  ns.println();
							  }
							  ns.print("first Match:");
							  for(int i=0;i<n;i++) {
								  ns.print(firstMatch[i]+" ");
							  }
							  ns.println();
							  ns.print("iMatch:");
							  for(int i=0;i<n;i++) {
								  ns.print(iMatch[i]);
							  }
							  ns.println();
							  ns.println("printing nm1s");
							  for(int i=0;i<nm1s.size();i++) {  //print nm1s
								  List<int[]> debug = nm1s.get(i);
								  for(int j=0;j<debug.size();j++) {
									  ns.print(debug.get(j)[0]+" "+debug.get(j)[1]+",");
								  }
								  ns.println();
							  }
							  
							  */
							  stable = false;
							  break iter;
						  }

						  ////////////////////////////////////////////////////////////////////////////////////////////////////////////      setup      ////////////////////////
						  
						  boolean reg = false; //nm1 starts with r
						  if(a==1)
							  reg = true; //nm1 starts with l (the regular way)
						  boolean total = false; //nm1 decided with l+r (inefficient but very fair)
						  int nm2type = 1; //0=original PII, 1=diagonal, 2=random
						  boolean nm2switch = false; //if two nm2's, pick better pair (slight speedup)

						  //init everything
						  int[] nm1i = new int[n]; //nm1[i]=j => (i,j) is nm1
						  int[] nm1j = new int[n]; //nm1[j] = i
						  int[] nm2geni = new int[n]; //etc
						  int[] nm2genj = new int[n];
						  int[] nm2 = new int[n];

						  for(int i=0;i<n;i++) {
							  nm1i[i] = -1;
							  nm1j[i] = -1;
							  nm2geni[i] = -1;
							  nm2genj[i] = -1;
							  nm2[i] = -1;
						  }

						  if(stepNum==0) {   //to see initial match in case of error
							  for(int i=0;i<n;i++) {
								  firstMatch[i] = iMatch[i];
							  }
						  }
						  //now finally the PII algorithm starts

						  //graph[i][j][l r type (0=reg,1=match,2=nm1gen)]
						  //find nm1gen: for every row, find the lowest l/r/sum that's unstable
						  stable = true;
						  for(int i=0;i<n;i++) {
							  int lowest = 2*n+1; //sentinel
							  int ref = -1;
							  for(int j=0;j<n;j++) {
								  if(graph[i][j][2] != 1 && reg) //update nm1gen from last time
									  graph[i][j][2] = 0;
								  else if(graph[j][i][2] != 1 && !reg)
									  graph[j][i][2] = 0;

								  graph[i][j][3] = 1; //initialize valids (used for some nm2 types)
								  if(graph[i][j][0] < graph[i][iMatch[i]][0] &&  //if unstable (the regular way)
										  graph[i][j][1] < graph[jMatch[j]][j][1] && reg) {
									  stable = false;
									  if(graph[i][j][0] < lowest && !total) { //update lowest
										  lowest = graph[i][j][0];
										  ref = j;
									  }
									  else if((graph[i][j][0] + graph[i][j][1]) < lowest && total) { //test using l+r instead of l
										  lowest = graph[i][j][0] + graph[i][j][1];
										  ref = j;
									  }
								  }
								  else if(graph[j][i][0] < graph[j][iMatch[j]][0] && //test with r value first
										  graph[j][i][1] < graph[jMatch[i]][i][1] && !reg) {
									  stable = false;
									  if(graph[j][i][1] < lowest && !total) {
										  lowest = graph[j][i][1];
										  ref = j;
									  }
									  else if((graph[j][i][0] + graph[j][i][1]) < lowest && total) { // l+r
										  lowest = graph[j][i][0] + graph[j][i][1];
										  ref = j;
									  }
								  }
							  }
							  if(ref != -1 && reg)  // set nm1gen
								  graph[i][ref][2] = 2;
							  else if(ref != -1 && !reg)
								  graph[ref][i][2] = 2;
						  } //done finding nm1gen

						  if(stable)
							  break; //exit the iter loop

						  //find nm1: for every column, find the lowest r that's nm1gen
						  for(int j=0;j<n;j++) {
							  int lowest = 2*n+1; //sentinel
							  int ref = -1;
							  for(int i=0;i<n;i++) {
								  if(graph[i][j][2] == 2 && reg) { //if nm1gen
									  if(graph[i][j][1] < lowest && !total) { //reg nm1
										  lowest = graph[i][j][1];
										  ref = i;
									  }
									  else if((graph[i][j][0] + graph[i][j][1]) < lowest && total) { // use l+r instead
										  lowest = graph[i][j][0] + graph[i][j][1];
										  ref = i;
									  }
								  }
								  else if(graph[j][i][2] == 2 && !reg) { //go by rows first
									  if(graph[j][i][0] < lowest && !total) {
										  lowest = graph[j][i][0];
										  ref = i;
									  }
									  else if((graph[j][i][0] + graph[j][i][1]) < lowest && total) {
										  lowest = graph[j][i][0] + graph[j][i][1];
										  ref = i;
									  }
								  }
							  }
							  if(ref != -1 && reg) { //update nm1
								  nm1i[ref] = j;
								  nm1j[j] = ref;
							  }
							  else if(ref != -1 && !reg) {
								  nm1i[j] = ref;
								  nm1j[ref] = j;
							  }
						  } //now we've found nm1

						  stepNum++;

						  //update the list of nm1 chains when its cycling (this section is stable match from cycle code)
						  if(past3 && !matchFromCycle && cycDetect) { //if it's past 3n iterations (guaranteed to cycle)
							  for(int i=0;i<n;i++) {
								  if(nm1i[i] != -1 && !repeats[i][nm1i[i]]) { //for every new nm1i
									  repeats[i][nm1i[i]] = true;
									  boolean added = false;
									  for(int j=0;j<nm1s.size();j++) { //find out where it goes
										  List<int[]> nm1Check = nm1s.get(j);
										  //System.out.println("testing "+nm1Check.get(nm1Check.size()-1)[0]+" "+nm1Check.get(nm1Check.size()-1)[1]);
										  if((nm1Check.get(nm1Check.size()-1)[0] == i || nm1Check.get(nm1Check.size()-1)[1] == nm1i[i] //if i or j matches an existing nm1
												  ) && iMatch[nm1Check.get(nm1Check.size()-1)[0]] == nm1Check.get(nm1Check.size()-1)[1]) { //and the nm1 is a current match
											  int[] toAdd = {i,nm1i[i]};
											  nm1Check.add(toAdd);
											  //System.out.println("adding "+toAdd[0]+" "+toAdd[1]+" to "+j);
											  if(nm1Check.size() > 2 && (nm1Check.get(nm1Check.size()-2)[0] == i && nm1Check.get(nm1Check.size()-3)[0] == i ||
													  nm1Check.get(nm1Check.size()-2)[1] == nm1i[i] && nm1Check.get(nm1Check.size()-3)[1] == nm1i[i])) { //if 3 in a row
												  parity.remove(j);
												  parity.add(j,nm1Check.size()%2);  //update parity to pick the middle nm1 in the chain of 3
												  //System.out.println("switching parity to "+nm1Check.size()%2+" for i "+i+" nm1i "+nm1i[i]+" j "+j);
											  }
											  added = true;
											  break;
										  }
									  }
									  if(!added) { //if it didn't go into an existing chain, start a new one
										  List<int[]> nm1 = new ArrayList<int[]>();
										  int[] toAdd = {i,nm1i[i]};
										  nm1.add(toAdd);
										  nm1s.add(nm1);
										  parity.add(-1);
										  //System.out.println("new "+toAdd[0]+" "+toAdd[1]+" to "+(nm1s.size()-1));
									  }
								  }
							  }
						  } //done with the nm1 chain code

						  for(int i=0;i<n;i++) {   //find nm2gen
							  if(nm1i[i] != -1) { //nm1i is at (i,nm1i[i])
								  nm2geni[jMatch[nm1i[i]]] = iMatch[i];
								  graph[i][iMatch[i]][2] = 0;  //these are no longer matches
								  graph[jMatch[nm1i[i]]][nm1i[i]][2] = 0;
							  }
						  } //found nm2gen

						  int nm2Count = 0;
						  if(nm2type==0) {  //this is the standard way to find nm2 pairs
							  for(int i=0;i<n;i++) {
								 if(nm2geni[i] != -1) { //if there's an nm2gen in row i
									 if(nm1i[i] == -1) { //if there's no nm1 in row i
										 if(nm1j[nm2geni[i]] == -1) { //if there's no nm1 in the same col
											 nm2[i] = nm2geni[i]; //isolated node
											 nm2Count++;
										 }
										 else {  //then it's a row end
											 int iPlus = i;
											 int jPlus = nm2geni[i];
											 while(true) { //find the col end
												 iPlus = jMatch[jPlus]; //i+,j+ is the match in the same col as the nm2gen
												 jPlus = nm2geni[iPlus]; //i+,j+ is the next nm2gen
												 if(nm1j[jPlus] == -1)
													 break; //found the col end
											 }
											 nm2[i] = jPlus;
											 nm2Count++;
										 }
									 }
								 }
							  } //now we've found nm2
						  }
						  else {  //the other two nm2types start by calculating invalids
							  for(int i=0;i<n;i++) {
								  for(int j=0;j<n;j++) {
									  if(nm1i[i] != -1 || nm1j[j] != -1) { //if there's an nm1 in your row/col
										  graph[i][j][3] = 0; //not valid
										  if(iMatch[i] == j) { //no longer a match
											  iMatch[i] = -1;
											  jMatch[j] = -1;
										  }
									  }
								  }
							  }
							  for(int i=0;i<n;i++) {
								  for(int j=0;j<n;j++) {
									  if(iMatch[i] != -1 || jMatch[j] != -1) { //if there's a surviving match in your row/col
										  graph[i][j][3] = 0; //not valid
									  }
								  }
							  }
							  int jinit = 0;
							  int firsti = -1;
							  int firstj = -1;
							  for(int i=0;i<n;i++) { //make nm2 the diagonal
								  for(int j=jinit;j<n;j++) {
									  if(graph[i][j][3] == 1) {
										  if(firsti==-1) {
											  firsti=i;
											  firstj=j;
										  }
										  nm2[i] = j;
										  nm2Count++;
										  jinit = j+1;
										  break;
									  }
								  }
							  } //now we've found nm2 along diagonal
							  
							  if(firsti != -1 && nm2type==2) { //randomly generate nm2 from valids
								  int[] shuffled = new int[nm2Count];
								  int inc = 0;
								  for(int j=0;j<n;j++) {
									  if(graph[firsti][j][3] == 1) {
										  shuffled[inc] = j;
										  inc++;
									  }
								  } //now shuffled is a list of the j values
								  for(int i=0;i<nm2Count;i++) { //Knuth shuffle the array
									  int j = i+(int)(Math.random()*(nm2Count-i));
									  int holder = shuffled[i];
									  shuffled[i] = shuffled[j];
									  shuffled[j] = holder;
								  }
								  inc = 0;
								  for(int i=0;i<n;i++) { //now assign the random nm2s
									  if(graph[i][firstj][3] == 1) {
										  nm2[i] = shuffled[inc];
										  inc++;
									  }
								  }

							  } //found random nm2s
						  }

						  if(nm2Count == 2 && nm2switch) { //switches to better pair of nm2s
							  boolean first = true;
							  int ref1 = -1;
							  int ref2 = -1;
							  for(int i=0;i<n;i++) { //find the two nm2s
								  if(nm2[i] != -1 && first) {
									  first = false;
									  ref1 = i;
								  }
								  else if(nm2[i] != -1) {
									  ref2 = i;
								  }
							  }
							  //if the sum of l+r of the two current nm2s is worse than the other possible nm2s, switch
							  if(graph[ref1][nm2[ref2]][0]+graph[ref1][nm2[ref2]][1]+graph[ref2][nm2[ref1]][0]+
									  graph[ref2][nm2[ref1]][1]<graph[ref1][nm2[ref1]][0] + graph[ref1][nm2[ref1]][1]+
									  graph[ref2][nm2[ref2]][0] +graph[ref2][nm2[ref2]][1]) {
								  int holder = nm2[ref1];
								  nm2[ref1] = nm2[ref2];
								  nm2[ref2] = holder;
							  }
						  }

						  for(int i=0;i<n;i++) {  //update matches in graph and arrays
							  if(nm1i[i] != -1) {
								  graph[i][nm1i[i]][2] = 1;
								  iMatch[i] = nm1i[i];
								  jMatch[nm1i[i]] = i;
							  }
							  else if(nm2[i] != -1) {
								  graph[i][nm2[i]][2] = 1; //this is a new match
								  iMatch[i] = nm2[i];
								  jMatch[nm2[i]] = i;
							  }
						  } //now we've completed a step

						  if(init2 && s==2*n && a==0) //init2 runs the first loop 2n times
							  s=c*n+5; //set s to exit the a loop
					  } //end of step/iter

					  if(stable) {
						  
						  steps[a] += stepNum;
						  counts[a]++;

						  if(!cycled && stepNum > mostSteps[0]) //update mostSteps
							  mostSteps[0] = stepNum;
						  else if(cycled && stepNum > mostSteps[1])
							  mostSteps[1] = stepNum;

						  break; //break in case a==0
					  }
					  else { //if unstable
						  //System.out.println("found unstable");
						  //if(a ==1)
							  //ps.println("found unstable on trial" + t);
					  }
				} //end a loop
				count++;
			} //end t loop
			System.out.println("finished n="+n+" , longest cycle "+longest+" most steps acyclic "+(mostSteps[0]/n)+" cyclic "+(mostSteps[1]/n));
			ps.println("n="+n+" trials:"+count+" "); //print out info specific to n value
			ps.println("together:"+(counts[0]+counts[1]));
			togTotal = togTotal + counts[0] + counts[1];
			for(int i=0;i<test;i++) { //print out successes in each phase
				ps.println(i+":"+counts[i]+" "+((0.0+steps[i])/counts[i]));
			}
			for(int i=0;i<test;i++) { //update the total counts
				totalSteps[i] += steps[i];
				totals[i] += counts[i];
			}
			countTotal += count;
		} //end n loop
		System.out.println("done");
		ps.println();
		ps.println("total:"+countTotal+" "); //print out totals
		ps.println("together:"+togTotal);
		for(int i=0;i<test;i++) {
			ps.println(i+":"+totals[i]+" "+((0.0+totalSteps[i])/totals[i]));
		}
		PrintStream cm = new PrintStream(new FileOutputStream("finished.txt")); //so I know when it finished
		cm.print("done");
	} //end main

	public static int[][][] getInit(int n) throws FileNotFoundException { //read the preference list and initial match from init.txt
		Scanner sc = new Scanner(new FileReader("init.txt"));
		int[][][] graph = new int[n][n][4];
		while(sc.hasNext()) {
			for(int i=0;i<n;i++) {
				for(int j=0;j<n;j++) {
					graph[i][j][0] = sc.nextInt();
					graph[i][j][1] = sc.nextInt();
					//graph[i][j][2] = sc.nextInt();
					//graph[i][j][3] = sc.nextInt();
				}
			}
		}
		return graph;
	}

	public static int[][][] getHadoop(int n) throws FileNotFoundException { //read the preference list and initial match from part-00000
		Scanner sc = new Scanner(new FileReader("part-00000"));
		int[][][] graph = new int[n][n][4];
		while(sc.hasNext()) {
			int i = sc.nextInt();
			int j = sc.nextInt();
			String[] data = sc.next().split(",");
			if(data[0].equals("true"))
				graph[i][j][2] = 1;
			else
				graph[i][j][2] = 0;
			graph[i][j][0] = Integer.parseInt(data[1]);
			graph[i][j][1] = Integer.parseInt(data[2]);
		}
		return graph;
	}

	public static void print(int[][][] graph,int n,String file) throws FileNotFoundException { //print out the graph
		PrintStream out = new PrintStream(new FileOutputStream(file));

		for(int i=0;i<n;i++) {
			for(int j=0;j<n;j++) {
				for(int k=0;k<4;k++) {
					out.print(graph[i][j][k]+" ");
				}
				out.print("\t");
			}
			out.println();
		}
		return;
	}

	public static int[][][] create(int n) throws FileNotFoundException { //creates a random preference list
		int[][] matrix = new int[n][n];
		int[][][] toReturn = new int[n][n][4];
		int[] array = new int[n];

		for(int j=0;j<n;j++) { //set up matrix
			for(int i=0;i<n;i++)
				matrix[i][j] = i+1; //used in generating a random match
		}

		for(int j=0;j<n;j++) { //Knuth shuffle matrix by column
			for(int i=0;i<n;i++) {
				int num = (int) (Math.random()*(n-i))+i; //pick random number i<num<n-1 to swap pointers with
				int k = matrix[num][j];
				int oldVal = matrix[i][j];
				matrix[num][j] = oldVal;
				matrix[i][j] = k;
			}
		}

		for(int i=0;i<n;i++) { //for every row
			for(int j=0;j<n;j++) //set up array
				array[j] = j+1;
			for(int j=0;j<n;j++) {	   //knuth shuffle array
				int num = (int) (Math.random()*(n-j))+j;
				int k = array[num];
				int oldVal = array[j];
				array[num] = oldVal;
				array[j] = k;
				toReturn[i][j][0] = array[j];
				toReturn[i][j][1] = matrix[i][j];
				toReturn[i][j][3] = j;
			}
		}
		return toReturn;
	}	//end create method

	//not formatted right
	  public static Object[] smartInit(int[][][] graph,int n,int r) {
		  //r=0 is regular, r=1 is transposed
		  //all l=1 broadcast to (k>i,j) that they can't be chosen
		  int[] iMatch = new int[n];
		  int[] jMatch = new int[n];
		  for(int i=0;i<n;i++) {
			  iMatch[i] = -1;
			  jMatch[i] = -1;
		  }
		  int count = 0;

		  for(int m=1;m<=n;m++) { //for all preferences
			  for(int i=0;i<n;i++) {
				  if(iMatch[i] == -1 && r == 0) { //if there's no pair in this row yet
					  for(int j=0;j<n;j++) {
						  if(graph[i][j][0] == m) {
							  boolean used = false;
							  for(int k=0;k<n;k++) {
								  if(iMatch[k]==j) { //if there's a pair already in this column
									  if(graph[i][j][0] == graph[k][j][0] && graph[i][j][1] < graph[k][j][1]) { //if this one is better
										  iMatch[k] = -1; //take out the worse one
										  jMatch[j] = -1;
										  count--;
									  }
									  else
										  used = true;
								  }
							  }
							  if(!used) { //if this column is free to use
								  iMatch[i] = j;
								  jMatch[j] = i;
								  count++;
							  }
							  break;
						  }
					  }
				  }
				  else if(jMatch[i] == -1 && r == 1) { //same thing, transposed
					  for(int j=0;j<n;j++) {
						  if(graph[j][i][1] == m) {
							  boolean used = false;
							  for(int k=0;k<n;k++) {
								  if(jMatch[k]==j) {
									  if(graph[j][i][1] == graph[j][k][1] && graph[j][i][0] < graph[j][k][0]) {
										  jMatch[k] = -1;
										  iMatch[j] = -1;
										  count--;
									  }
									  else
										  used = true;
								  }
							  }
							  if(!used) {
								  jMatch[i] = j;
								  iMatch[j] = i;
								  count++;
							  }
							  break;
						  }
					  }
				  }
			  }
			  if(count==n)
				  break;
		  }
		  return new Object[] {iMatch,jMatch,count};
	  }

	  public static Object[] galeShapley(int[][][] graph,int m,int n) { //GS algorithm
		  //m==0 is man optimal, m=1 is woman optimal
		  int[] iMatch = new int[n];
		  int[] jMatch = new int[n];
		  for(int i=0;i<n;i++) {
			  iMatch[i] = -1;
			  jMatch[i] = -1;
		  }
		  //graph: 0=l, 1=r, 2=0 if not tried

		  while(true) { //while an unpaired man exists
			  int i=-1;
			  for(int j=0;j<n;j++) { //find an unpaired man (or woman)
				  if(m==0 && iMatch[j] == -1) {
					  i=j;
					  break;
				  }
				  else if(m==1 && jMatch[j] == -1) {
					  i=j;
					  break;
				  }
			  }
			  if(i==-1) //if everybody is paired
				  break;
			  int j=-1;
			  int lowest = n+1; //sentinel
			  for(int k=0;k<n;k++) {   //if lowest so far and untried and (unpaired or better)
				  if(m==0 && graph[i][k][0]<lowest && (jMatch[k]==-1 || graph[jMatch[k]][k][1]>graph[i][k][1])) {
					  j=k;
					  lowest = graph[i][k][0];
				  }
				  else if(m==1 && graph[k][i][1]<lowest && (iMatch[k]==-1 || graph[k][iMatch[k]][0]>graph[k][i][0])) {
					  j=k;
					  lowest = graph[k][i][1];
				  }
			  }
			  if(m==0 && jMatch[j] != -1) //update matches
				  iMatch[jMatch[j]] = -1;
			  else if(m==1 && iMatch[j] != -1)
				  jMatch[iMatch[j]] = -1;
			  if(m==0) {
				  iMatch[i] = j;
				  jMatch[j] = i;
			  }
			  else if(m==1) {
				  iMatch[j] = i;
				  jMatch[i] = j;
			  }
		  }
		  return new Object[] {iMatch,jMatch};
	  }

	  public static boolean check(int[][][] graph,int[] iMatches,int[] jMatches,int n) { //checks if it's a stable match

		  boolean stable = true;
		  for(int i=0;i<n;i++) { //check to make sure there are no unstable pairs
			  for(int j=0;j<n;j++) {
				  if(iMatches[i] != j) { //if it's not a match
					  if(graph[i][j][0] < graph[i][iMatches[i]][0]
							  && graph[i][j][1] < graph[jMatches[j]][j][1]) {
						  stable = false;
					  }
				  }
			  }
		  }
		  if(!stable){
			  //System.out.println("unstable");
		  }
		  return stable;
	  }
}
