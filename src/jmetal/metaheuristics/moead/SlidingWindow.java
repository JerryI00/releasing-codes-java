package jmetal.metaheuristics.moead;

import java.io.*;
import java.util.*;

import jmetal.util.Configuration;

public class SlidingWindow implements Serializable {
	protected List<Window> windowList_;

	private int capacity_ = 0;

	public SlidingWindow() {
		windowList_ = new ArrayList<Window>();
	}

	public SlidingWindow(int maximumSize) {
		windowList_ = new ArrayList<Window>();
		capacity_ = maximumSize;
	}

	public boolean add(Window element) {
		if (windowList_.size() == capacity_) {
			Configuration.logger_.severe("The population is full");
			Configuration.logger_.severe("Capacity is : " + capacity_);
			Configuration.logger_.severe("\t Size is: " + this.size());
			return false;
		} // if

		windowList_.add(element);
		return true;
	} // add

	public Window get(int i) {
		if (i >= windowList_.size()) {
			throw new IndexOutOfBoundsException("Index out of Bound " + i);
		}
		return windowList_.get(i);
	}

	public int getMaxSize() {
		return capacity_;
	} // getMaxSize

	public int size() {
		return windowList_.size();
	} // size

	public void clear() {
		windowList_.clear();
	} // clear

	public void remove(int position) {
		windowList_.remove(position);
	}

	public void readSlidingWindow(double[][] slidingWindow_) {
		int i;

		for (i = 0; i < size(); i++) {
			slidingWindow_[0][i] = get(i).getIndex();
			slidingWindow_[1][i] = get(i).getFitness();
		}
	}
}
