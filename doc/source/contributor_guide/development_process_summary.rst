.. _contributor_guide.development_process_summary:

===========================
Development Process Summary
===========================

Fork
----

Go to https://github.com/DassHydro/smash/fork and create your own ``Fork`` of the project.

Clone
-----

Clone the Git repository:

.. code-block:: none

    git clone https://github.com/username/smash.git

If you prefer working on the Git repository without authentication required, you can create a personal access token by following these
`detailed instructions <https://docs.github.com/en/authentication/keeping-your-account-and-data-secure/creating-a-personal-access-token>`__.
Once you have generated a token, clone the Git repository using the following command:

.. code-block:: none

    git clone https://username:token@github.com/username/smash.git

Replace ``username`` with your GitHub username and ``token`` with the personal access token you generated.

Install developer environment
-----------------------------

Go to the smash repository:

.. code-block:: none

    cd smash

It is recommended that you install the smash development environment using `Anaconda <https://www.anaconda.com/>`__:

.. code-block:: none

    conda env create -f environment-dev.yml

Activate the conda environment:

.. code-block:: none

    conda activate smash-dev

Develop contribution
--------------------

Create a new branch, based on main, on which you will implement your development:

.. code-block:: none

    (smash-dev) git checkout -b your-development-branch

Then, build in editable mode:

.. code-block:: none

    (smash-dev) make edit

Submit
------

Before submitting your changes, make sure that the unit tests pass (although they will also be executed automatically once the submission has been made):

.. code-block:: none

    (smash-dev) make test

As well as linters and formatters:

.. code-block:: none

    (smash-dev) make check
    (smash-dev) make format

If changes have also been made to the documentation, also run the command to check the compilation of the documentation:

.. code-block:: none

    (smash-dev) make doc

If all the tests pass, commit your changes:

.. code-block:: none

    (smash-dev) git add .
    (smash-dev) git commit

Write a clear message to help reviewers understand what has been done in this contribution and finally, push your changes back to your fork:

.. code-block:: none

    (smash-dev) git push --set-upstream origin your-development-branch

You will be asked for your username and password (unless you have generated a personal token access). Then, go to GitHub.
The new branch will show up with a green ``Pull Request`` button. Make sure the title and message are clear. Then click the button to submit it.

Review process
--------------

Reviewers (the other developers and interested community members) will write inline and/or general comments on your Pull Request (``PR``) to help you improve its implementation, 
documentation and style.

To update your ``PR``, make your changes on your local repository, commit, run tests, and only if they succeed push to your fork. 
As soon as those changes are pushed up (to the same branch as before) the ``PR`` will update automatically. 
If you have no idea how to fix the test failures, you may push your changes anyway and ask for help in a ``PR`` comment.

Various continuous integration (``CI``) services are triggered after each ``PR`` update to build the code, run unit tests, compare adjoint code and check documentation compilation. 
The ``CI`` tests must pass before your ``PR`` can be merged. If ``CI`` fails, you can find out why by clicking on the “failed” icon (red cross) and inspecting the build and test log. 
