---
title: Overview:
notebook: Final_Project_8_10.ipynb
nav_include: 1
---

## CS109A Data Science Final Project
Jinwon Chung and Kate Zhou
Group #XX

### 1. Motivation

LendingClub is the first peer-to-peer lending network to register with the Securities and Exchange Commission(SEC). The LendingClub platform connects borrowers to investors and facilitates the payment of loans ranging from 40,000.

The basic idea is that borrowers apply for a loan on the LendingClub Marketplace. Investors can review the loan applications, along with risk analysis provided by LendingClub, to determine how much of the loan request they are willing to fund. When the loan is fully funded, the borrower will receive the loan. Investors can also choose to have loans selected manually or automatically via a specified strategy. Loan details such as grade or subgrade, loan purpose, interest rate, and borrower information can all be used to evaluate a loan for possible investment.

As part of this marketplace, LendingClub makes a significant amount of the loan data that is available to investors available online. Investors can make their own investment decisions and strategies based on the information provided by LendingClub.
As with other types of loan and investment platforms, Lending Club is the site of possible discrimination or unfair lending practices. Note that Lending Club is an \"Equal Housing Lender\" (logo displayed in the bottom of each page of the website) which means that \"the bank makes loans without regard to race, color, religion, national origin, sex, handicap, or familial status.\" Such policies in the past have failed to stop discrimination, most publicly against loan applicants of color, especially African American applicants. Racial discrimination has continued in new forms following the federal Fair Housing Act; investors have adapted their approach to fit within the guidelines of the law while maintaining results with damaging and immoral effects on communities of color. Other systems do not explicitly discriminate against minorities but enable practices such as refusing loans from specific zip codes that in effect harm minority applicants. <br/> <br/>

### Project Objective: Create a model to choose Lending Club investments that maximize expected return.
The goal of the analysis was to choose a model that maximized the expected rate of return on investments. We considered the predictors that would be available to an investor at the time of purchase and which of them would be useful for our prediction. We considered which variables might be a good indicator of lifetime return. We also considered the risks and limitations of our predictions.
