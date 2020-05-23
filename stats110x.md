## A dump of Harvard stats course

### Week 5 concepts

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