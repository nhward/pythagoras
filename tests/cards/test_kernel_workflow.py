"""Kernel families are committed through parameters and executed by training."""
import pytest
from playwright.sync_api import expect
from shiny.pytest import create_app_fixture

app = create_app_fixture(app='../scenarios/kernel_workflow.py', scope='function')


@pytest.mark.ui
def test_gp_method_kernel_commitment_and_training(page, app):
    page.set_viewport_size({'width': 1800, 'height': 1400})
    page.goto(app.url)
    expect(page.locator('#cand_methods-Summary')).to_contain_text('Regression', timeout=30000)
    page.locator('#cand_methods-Card').hover()
    page.locator('#cand_methods-ExpandButton').click()
    table = page.locator('#cand_methods-Table')
    expect(table.get_by_text('Gaussian Process', exact=True)).to_be_visible(timeout=30000)
    table.get_by_role('checkbox', name='Select Gaussian Process', exact=True).check()
    page.locator('#cand_methods-Commit').click()
    expect(page.locator('#cand_methods-Status')).to_contain_text('Committed method', timeout=30000)
    page.locator('#cand_methods-Card').hover()
    page.locator('#cand_methods-ContractButton').click()
    page.locator('#cand_parameters-Card').hover()
    page.locator('#cand_parameters-ExpandButton').click()
    parameter_table = page.locator('#cand_parameters-Table')
    expect(parameter_table).to_contain_text('Kernel family', timeout=30000)
    parameter_table.get_by_text('Kernel family', exact=True).locator('xpath=ancestor::tr').get_by_role('checkbox').check()
    expect(page.locator('#cand_parameters-Status')).to_contain_text('3 possible combinations')
    page.locator('#cand_parameters-Search').fill('Polynomial degree')
    expect(parameter_table.locator('tbody tr')).to_have_count(1)
    parameter_table.get_by_text('Polynomial degree', exact=True).locator('xpath=ancestor::tr').get_by_role('checkbox').check()
    page.locator('#cand_parameters-Commit').click()
    expect(page.locator('#KernelSpec')).to_have_text('5', timeout=30000)
    page.locator('#cand_parameters-Card').hover()
    page.locator('#cand_parameters-ContractButton').click()
    expect(page.locator('#cand_train-Progress')).to_contain_text('Ready', timeout=30000)
    page.locator('#cand_train-Start').click()
    expect(page.locator('#cand_train-Progress')).to_contain_text('Finished', timeout=45000)
    expect(page.locator('#KernelTrials')).to_have_text('linear;poly;rbf', timeout=30000)
