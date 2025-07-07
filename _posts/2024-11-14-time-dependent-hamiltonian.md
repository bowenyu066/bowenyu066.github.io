---
title: On the Explicit Expression of a Time-dependent Hamiltonian in Heisenberg Picture
date: 2024-11-14 11:28:06
permalink: /posts/notes/time-dependent-hamiltonian/
excerpt: "We prove the explicit expression of the time-dependent Hamiltonian in Heisenberg picture in this post."
tags: 
    - Physics
categories:
    - Notes
---

The Heisenberg Hamiltonian

$$
\hat{H}_H\left(t,t_0\right)=U^\dagger \left(t,t_0\right) \hat{H}_S\left(t\right) U(t,t_0)
$$

We'll prove the following **Theorem**.

## Statement of the Theorem

$$
\hat{H}_H\left(t,t_0\right)=\hat{H}_S\left(t\right) + \sum_{n=1}^{\infty} \left(-\frac{i}{\hbar}\right)^n \int_{t_0}^{t}dt_1 \int_{t_0}^{t_1}dt_2  \cdots \int_{t_0}^{t_{n-1}}dt_n \ \hat{C}_n\left(t,t_1,t_2,\cdots,t_n\right)
$$

where

$$
\hat{C}_n\left(t,t_1,t_2,\cdots,t_n\right) = \left[\cdots \left[\left[\hat{H}_S\left(t\right),\hat{H}_S\left(t_1\right) \right],\hat{H}_S\left(t_2\right)\right] \cdots, \hat{H}_S\left(t_n\right) \right]
$$

and it can be easily seen that

$$
\hat{C}_n\left(t,t_1,t_2,\cdots,t_n\right) = \left[\hat{C}_{n-1}\left(t,t_1,t_2,\cdots,t_{n-1}\right),\hat{H}_S\left(t_n\right) \right]
$$

To deal with the problem, we first sketch the proofs of several lemmas.

## Lemma 1

### Statement of Lemma 1

$$
\int_{t_0}^{t}dt_1 \int_{t_0}^{t_1}dt_2  \cdots \int_{t_0}^{t_{n-1}}dt_n = \int_{t_0}^{t}dt_n \int_{t_{n}}^{t}dt_{n-1}  \cdots \int_{t_2}^{t}dt_1
$$

### Proof of Lemma 1

*Proof:* By induction.

- Starting point:

$$
\int_{t_0}^{t}dt_1 \int_{t_0}^{t_1}dt_2 = \int_{t_0}^{t}dt_2 \int_{t_2}^{t}dt_{1}
$$

where LHS and RHS both correspond to the integration over

$$
D=\{(t_1,t_2)\ | \ t_0\leq t_1 \leq t,\ t_0\leq t_2 \leq t, \ t_2 \leq t_1\}
$$

- Induction: suppose we already have

$$
\int_{t_0}^{t}dt_1 \int_{t_0}^{t_1}dt_2  \cdots \int_{t_0}^{t_{n-1}}dt_n = \int_{t_0}^{t}dt_n \int_{t_{n}}^{t}dt_{n-1}  \cdots \int_{t_2}^{t}dt_1
$$

Then

$$
\begin{align*}
    & \int_{t_0}^{t}dt_1 \int_{t_0}^{t_1}dt_2  \cdots \int_{t_0}^{t_{n-1}}dt_n \int_{t_0}^{t_{n}}dt_{n+1}\\ =& \boxed {\int_{t_0}^{t}dt_1 \int_{t_0}^{t_1}dt_{n+1}} \int_{t_{n+1}}^{t_1}dt_n  \cdots \int_{t_3}^{t}dt_2 
    \\ =& \int_{t_0}^{t}dt_{n+1} \boxed{ \int_{t_{n+1}}^{t}dt_{1} \int_{t_{n+1}}^{t_1}dt_n } \cdots \int_{t_3}^{t}dt_2 
    \\ =& \cdots
    \\ =& \int_{t_0}^{t}dt_{n+1} \int_{t_{n+1}}^{t}dt_{n}  \cdots \int_{t_3}^{t}dt_2  \int_{t_2}^{t}dt_1 \quad \quad \quad \Box
\end{align*} 
$$

## Lemma 2

### Statement of Lemma 2

$$
\frac{\partial}{\partial t_0} U\left(t, t_0\right) = \frac{i}{\hbar} U\left(t, t_0\right) H_S (t_0)
$$

### Proof of Lemma 2

*Proof:* From

$$
U\left(t, t_0\right) = \mathbb{1} + \sum_{n=1}^{\infty} \left(-\frac{i}{\hbar}\right)^n \int_{t_0}^{t}dt_1 H_S(t_1) \int_{t_0}^{t_1}dt_2 H_S(t_2) \cdots \int_{t_0}^{t_{n-1}}dt_n H_S(t_n)
$$

This is not hard to prove. (details omitted) \\(\Box\\)

## Lemma 3

### Statement of Lemma 3

$$
\frac{\partial}{\partial t_0} H_H\left(t, t_0\right) = \frac{i}{\hbar} \left[H_H\left(t, t_0\right), H_S (t_0)\right]
$$

### Proof of Lemma 3

$$
\frac{\partial}{\partial t_0} H_H\left(t, t_0\right) = \left(\frac{\partial}{\partial t_0}U^\dagger \left(t,t_0\right) \right) \hat{H}_S\left(t\right) U(t,t_0) +  U^\dagger(t,t_0) \hat{H}_S\left(t\right) \left(\frac{\partial}{\partial t_0}U \left(t,t_0\right)\right)
$$

From Lemma 2, \\(\displaystyle \frac{\partial}{\partial t_0} U^\dagger\left(t, t_0\right) = -\frac{i}{\hbar} H_S (t_0) U^\dagger\left(t, t_0\right)\\). Substitute them back to the previous equation, we finish the proof of Lemma 3. \\(\Box\\)

## Lemma 4

### Statement of Lemma 4

Let

$$
f(t,t_0) = \hat{H}_S\left(t\right) + \sum_{n=1}^{\infty} \left(-\frac{i}{\hbar}\right)^n \int_{t_0}^{t}dt_1 \int_{t_0}^{t_1}dt_2  \cdots \int_{t_0}^{t_{n-1}}dt_n \ \hat{C}_n\left(t,t_1,t_2,\cdots,t_n\right)
$$

We have

$$
\frac{\partial}{\partial t_0} f\left(t, t_0\right) = \frac{i}{\hbar} \left[f\left(t, t_0\right), H_S (t_0)\right]
$$

### Proof of Lemma 4

*Proof:* From Lemma 1,

$$
f(t,t_0) = \hat{H}_S\left(t\right) + \sum_{n=1}^{\infty} \left(-\frac{i}{\hbar}\right)^n \int_{t_0}^{t}dt_n \int_{t_{n}}^{t}dt_{n-1}  \cdots \int_{t_2}^{t}dt_1 \ \hat{C}_n\left(t,t_1,t_2,\cdots,t_n\right)
$$

Therefore,

$$
\frac{\partial}{\partial t_0} f(t,t_0) = - \sum_{n=1}^{\infty} \left(-\frac{i}{\hbar}\right)^n \int_{t_{n}}^{t}dt_{n-1}  \cdots \int_{t_2}^{t}dt_1 \ \hat{C}_n\left(t,t_1,t_2,\cdots,t_{n-1},t_0\right)
$$

Rewrite

$$
\hat{C}_n\left(t,t_1,t_2,\cdots,t_{n-1}, t_0\right) = \left[\hat{C}_{n-1}\left(t,t_1,t_2,\cdots,t_{n-1}\right),\hat{H}_S\left(t_0\right) \right]
$$

and notice

$$
\hat{C}_0\left(t\right) = \hat{H}_S\left(t\right)
$$

we get

$$
\begin{align*}
    \frac{\partial}{\partial t_0} f(t,t_0) &= \frac{i}{\hbar} \sum_{n=1}^{\infty} \left(-\frac{i}{\hbar}\right)^{n-1} \int_{t_{n}}^{t}dt_{n-1}  \cdots \int_{t_2}^{t}dt_1 \ \left[\hat{C}_{n-1}\left(t,t_1,t_2,\cdots,t_{n-1}\right),\hat{H}_S\left(t_0\right) \right] \\
    &=\frac{i}{\hbar} \left[f\left(t, t_0\right), H_S (t_0)\right]
\end{align*}
$$

This finishes our proof of Lemma 4. \\(\Box\\)

## Proof of the Theorem

Now we get back to the desired theorem.

*Proof of the Theorem:*

From Lemma 3 and Lemma 4,

$$
\frac{\partial}{\partial t_0} H_H\left(t, t_0\right) = \frac{i}{\hbar} \left[H_H\left(t, t_0\right), H_S (t_0)\right]
$$

$$
\frac{\partial}{\partial t_0} f\left(t, t_0\right) = \frac{i}{\hbar} \left[f\left(t, t_0\right), H_S (t_0)\right]
$$

which means \\(H_H\left(t, t_0\right)\\) and \\(f\left(t, t_0\right)\\) satisfies the same differential equation. 

On the other hand, 

$$
H_H\left(t, t_0=t\right) = f\left(t, t_0=t\right) = H_S(t)
$$

which means they share the same initial condition.

Thus, for any \\(t_0<t\\), we have

$$
H_H\left(t, t_0\right) = f\left(t, t_0\right)
$$

This marks the end of our proof. \\(\Box\\)

<script src="https://giscus.app/client.js"
        data-repo="bowenyu066/bowenyu066.github.io"
        data-repo-id="R_kgDOOSbJ2A"
        data-category="Announcements"
        data-category-id="DIC_kwDOOSbJ2M4CsmZz"
        data-mapping="pathname"
        data-strict="0"
        data-reactions-enabled="1"
        data-emit-metadata="0"
        data-input-position="bottom"
        data-theme="light"
        data-lang="en"
        crossorigin="anonymous"
        async>
</script>