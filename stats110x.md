## A dump of Harvard stats course

### Week 7 concepts

> Reversibility?

Q is the transition matrix of a Markov chain.

Suppose there is an **$s$** such that 
$s = (s_1,...,s_M)$ with $s_i\geq0, \sum_is_i=1$ and

$$s_iq_{ij}=s_jq_{ji}$$
for all states $i$ & $j$.
This is the reversibility condition

> Reversible implies stationary

1. If Q is symmetric, then the uniform distribution is stationary

2. If Q is doubly stochastic, then the uniform distribution is stationary

3. If markov chain is random walk on an undirected network then distribution is proportional to the degree sequence of the network

4. If Markov chain is birth-death chain, then the stationary distribution can be constructed such that $s_iq_{ij}=s_jq_{ji}$


```
A nonnegative matrix such that the row sums and the column sums are all equal to  1  is called a doubly stochastic matrix
```
> Google Page Rank mathematics

PageRank is the unique stationary distribution of the matrix $G = \alpha Q + (1-\alpha) \frac{1}{M}*J,$

[Page Rank Explanation - Cornell Math](http://pi.math.cornell.edu/~mec/Winter2009/RalucaRemus/Lecture3/lecture3.html)

> Stationary distribution

$sQ=s$

The marginal distribution does not change as we move forward in time

```NOTE:```
A Markov chain whose initial distribution is the stationary distribution  𝐬  will stay in the stationary distribution forever

### Convergence

If a markov chain is irreducible and aperiodic - Then 
$P(X_n = i)$ converges to $s_i$ as $n\to{inf}$

### Existence and Uniqueness

Finite state space - It exists

Irreducible markov chain - It is unique (Perron-Frobenius theorem)

> Irreducible Markov Chain is one in which you can go from any i to any j

All states are recurrent in an irreducible markov chain with finite state space 

> Classification of states

### Recurrent

In the long run, no state is eventually abandoned

![](https://courses.edx.org/assets/courseware/v1/00dd54b801f70b3bd954da8e252868f8/asset-v1:HarvardX+STAT110x+1T2020+type@asset+block/7_chain1.png)

### Transient

Number of returns to a transient state is geometric

![](https://courses.edx.org/assets/courseware/v1/bec46639700a9710a042b294ad3caf28/asset-v1:HarvardX+STAT110x+1T2020+type@asset+block/7_chain2.png)

> Marginal distribution of $X_n$

To find this, take the distribution of $X_0$ and matrix multiply with $Q^n$

Irreducible markov chain - If for every i,j in the transition matrix, i,j entry is non-zero for some finite n.

Markov chains

The  𝑛 -step transition probability from  𝑖  to  𝑗  is the probability of being at  𝑗  exactly  𝑛  steps after being at  𝑖 .

$q^{(n)}_{ij} = P(X_n = j|X_0 = i)$

> Transition Matrix

The matrix that gives the probability of going from state i to state j


### Week 6 concepts

> Useful properties of conditional expectation

* If  𝑋  and  𝑌  are independent, then  𝐸(𝑌|𝑋)=𝐸(𝑌) 
* For any function  ℎ ,  𝐸(ℎ(𝑋)𝑌|𝑋)=ℎ(𝑋)𝐸(𝑌|𝑋)
* $𝐸(𝑌1+𝑌2|𝑋)=𝐸(𝑌1|𝑋)+𝐸(𝑌2|𝑋)$ , and 𝐸(𝑐𝑌|𝑋)=𝑐𝐸(𝑌|𝑋) for 𝑐 a constant (the latter is a special case of taking out what's known)
* ```Adams Law``` $E(E(Y|X)) = E(Y)$
*

### Conditional expectation given a random variable

In the example below, the expectation of a random variable given a random variable is a random variable!

$$E(Y|X) = X/2$$

> Conditional expectation

Analogous to conditional expectation

$$E(Y) = \sum_{i=1}^n E(Y|A_i) P(A_i)$$

> Conditional expectation given an event
Conditional expectation 𝐸(𝑌|𝐴) given an {event}: let 𝑌 be an r.v., and 𝐴 be an event. If we learn that 𝐴 occurred, our updated expectation for 𝑌, 𝐸(𝑌|𝐴), is computed analogously to 𝐸(𝑌), except using conditional probabilities given 𝐴.

>Conditional expectation given a random variable
Conditional expectation 𝐸(𝑌|𝑋) given a {random variable}: a more subtle question is how to define 𝐸(𝑌|𝑋), where 𝑋 and 𝑌 are both r.v.s. Intuitively, 𝐸(𝑌|𝑋) is the r.v. that best predicts 𝑌 using only the information available from  𝑋

> It must be normal?

It can be shown that the independence of the sum and difference is a unique characteristic of the Normal! That is, if  𝑋  and  𝑌  are i.i.d. and  𝑋+𝑌  is independent of  𝑋−𝑌 , then  𝑋  and  𝑌  must have Normal distributions.


>Multivariate Normal distribution

Why?
```
𝑍,𝑊∼i.i.d N(0,1) ,  (𝑍,𝑊)  is Bivariate Normal because the sum of independent Normals is Normal
```

> Correlation
$$$$

> Continuous case Joint PDF

```
If 𝑋 and 𝑌 are continuous with joint CDF 𝐹𝑋,𝑌, their joint PDF is the derivative of the joint CDF with respect to 𝑥 and 𝑦:
𝑓𝑋,𝑌(𝑥,𝑦)=∂2∂𝑥∂𝑦𝐹𝑋,𝑌(𝑥,𝑦)
```

> Law of total probability revisted



But if we only know the marginal PMFs of  𝑋  and  𝑌 , there is no way to recover the joint PMF without further assumptions.

* Joint Distribution
```
The joint PMF of discrete r.v.s  𝑋  and  𝑌  is the function  𝑝𝑋,𝑌  given by
𝑝𝑋,𝑌(𝑥,𝑦)=𝑃(𝑋=𝑥,𝑌=𝑦). 
The joint PMF of  𝑛  discrete r.v.s is defined analogously
```

* Marginal distribution
```
For discrete r.v.s 𝑋 and 𝑌, the marginal PMF of 𝑋 is
𝑃(𝑋=𝑥)=∑𝑦𝑃(𝑋=𝑥,𝑌=𝑦).
The marginal PMF of 𝑋 is the PMF of 𝑋, viewing 𝑋 individually rather than jointly with 𝑌. The above equation follows from the axioms of probability (we are summing over disjoint cases). The operation of summing over the possible values of 𝑌 in order to convert the joint PMF into the marginal PMF of 𝑋 is known as marginalizing out  𝑌
```
* Conditional distribution
* Covariance
* Correlation
* Multivariate Normal distribution
* Adam's Law
* Eve's Law


## Joint distribution
> Also called multivariate 

### Week 5 concepts

> What is the memory less property?



> What is the mean and the variance of a standard normal distribution?

[Just integrate baby](https://courses.edx.org/courses/course-v1:HarvardX+STAT110x+1T2020/courseware/28ff19f7be634e71a4aea103653b0608/9c707d053d854a98b3cc43e1353f7d32/?activate_block_id=block-v1%3AHarvardX%2BSTAT110x%2B1T2020%2Btype%40sequential%2Bblock%409c707d053d854a98b3cc43e1353f7d32)

> What is a poisson approximation?

If you have many events, each with a minute probability, and weakly dependent, then the sum of those events can be approximated to Pois(𝜆)






Law of the Unconscious Statistician
```
Theorem 5.5.1 (LOTUS).
If X is a discrete r.v. and g is a function from R to R, then
E(g(X))= ∑x g(x)P(X=x),
where the sum is taken over all possible values of X
```

```
There is a one-to-one correspondence between events and indicator r.v.s, and the probability of an event 𝐴 is the expected value of its indicator r.v. 𝐼𝐴:
𝑃(𝐴)=𝐸(𝐼𝐴)
```

### Week 4 concepts

> What is sum of independent normals?

It's Normal with mean as sum of means and std deviation as sum of std deviation squared

> What is universality of the uniform?

[The F of a distribution whose CDF is F is Uniform](https://courses.edx.org/courses/course-v1:HarvardX+STAT110x+1T2020/courseware/b3ac05d0c5324025b4a87f83b40d9738/b2cdb211988d4ae2a9406ad99a4ffac6/?child=first)

> What is location-scaling?

$$Y = \sigma*X + \mu$$ 

NOTE:

```
When using location-scale transformations, the shifting and scaling should be applied to the random variables themselves, not to their PDFs. For example, let U∼Unif(0,1), so the PDF f has f(x)=1 on (0,1) (and f(x)=0 elsewhere). Then 3U+1∼Unif(1,4), but 3f+1 is the function that equals 4 on (0,1) and 1 elsewhere, which is not a valid PDF since it does not integrate to 1 
```

$$P(a <  X \leq b) = F(b) - F(a) = \int_a^b f(x) dx.$$

What is connection between CDF and PDF?

```
the CDF is the accumulated area under the PDF
```

```
An r.v. has a continuous distribution if its CDF is differentiable. We also allow there to be endpoints (or finitely many points) where the CDF is continuous but not differentiable, as long as the CDF is differentiable everywhere else. A continuous random variable is a random variable with a continuous distribution
```


### Week 3 concepts


> Independence of random variables

[Useful definitions](https://courses.edx.org/courses/course-v1:HarvardX+STAT110x+1T2020/courseware/d7d4eef7f8d5461aa02ce3a490b9ff11/ed3436697f7c4c41b9ac2f44f689d5c9/3?activate_block_id=block-v1%3AHarvardX%2BSTAT110x%2B1T2020%2Btype%40html%2Bblock%40bb3adc990e6e489fa29cea4cd0764e72)

>I.I.D

We will often work with random variables that are independent and have the same distribution. We call such r.v.s independent and identically distributed, or i.i.d. for short

>Difference between random variable & distribution

''The word is not the thing; the map is not the territory.'' - Alfred Korzybski

>Categorical errors

To help avoid being categorically wrong, always think about what category an answer should have

> Cumulative distribution function

```
The cumulative distribution function (CDF) of an r.v. X is the function FX given by FX(x)=P(X≤x). When there is no risk of ambiguity, we sometimes drop the subscript and just write F (or some other letter) for a CDF
```

> Bernoulli distribution

```
An r.v.  𝑋  is said to have the Bernoulli distribution with parameter  𝑝  if  𝑃(𝑋=1)=𝑝  and  𝑃(𝑋=0)=1−𝑝 , where  0<𝑝<1 . We write this as  𝑋∼Bern(𝑝) . The symbol  ∼  is read ''is distributed as''.
```

Explanation of Random variable

x axis - np.unique
y axis - df['X'].value_counts()


``` 
Knowing the PMF of a discrete r.v. determines its distribution.
```


1. Bernoulli distribution
2. Binomial distribution
3. Hypergeometric distribution

### Week 2 concepts

```
Independence is completely different from disjointness. If  𝐴  and  𝐵  are disjoint, then  𝑃(𝐴∩𝐵)=0 , so disjoint events can be independent only if  𝑃(𝐴)=0  or  𝑃(𝐵)=0 . Knowing that  𝐴  occurs tells us that  𝐵  definitely did not occur, so  𝐴  clearly conveys information about  𝐵
```


> Conditional independence does not imply independence
```
To state this formally, let  𝐹  be the event that we've chosen the fair coin, and let  𝐴1  and  𝐴2  be the events that the first and second coin tosses land heads. Conditional on  𝐹 ,  𝐴1  and  𝐴2  are independent, but  𝐴1  and  𝐴2  are not unconditionally independent because  𝐴1  provides information about  𝐴2 .
```



### Week 1 concepts

Sampling with replacement

Sampling without replacement

Permutations

Birthday problem


[Course syllabus](https://courses.edx.org/courses/course-v1:HarvardX+STAT110x+1T2020/12b16231c4c94b8b994adfdd44d11b97/)

[Harvard main web page ](https://projects.iq.harvard.edu/stat110)


[Latex notation](https://courses.edx.org/courses/course-v1:HarvardX+STAT110x+1T2020/courseware/1450d9b731444cb1879e9ae02f1a7cd2/f75bb8ffa8784f18a1838e723c63c1f8/6?activate_block_id=block-v1%3AHarvardX%2BSTAT110x%2B1T2020%2Btype%40html%2Bblock%40c916494eacd54fc8a5a4713e9ecac484)


#### Bothering questions

1. What is the fundamental bridge?

2. What is the Vandermod identity?

3. Geometric random variable only takes finite values?

4. What is the Markov property?

5. What is a random walk?

6. What is an undirected network?
