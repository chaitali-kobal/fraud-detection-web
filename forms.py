from flask_wtf import FlaskForm
from wtforms import SelectField, DecimalField
from wtforms.validators import DataRequired, NumberRange

class TransactionForm(FlaskForm):
    transaction_type = SelectField(
        'Transaction Type',
        choices=[
            ('', 'Select Type'),
            ('PAYMENT', 'PAYMENT'),
            ('TRANSFER', 'TRANSFER'),
            ('CASH_OUT', 'CASH_OUT'),
            ('DEPOSIT', 'DEPOSIT'),
            ('DEBIT', 'DEBIT')
        ],
        validators=[DataRequired()]
    )
    
    amount = DecimalField(
        'Transaction Amount',
        validators=[DataRequired(), NumberRange(min=0.01)],
        places=2
    )
    
    sender_old_balance = DecimalField(
        'Sender\'s Old Balance',
        validators=[DataRequired(), NumberRange(min=0)],
        places=2
    )
    
    sender_new_balance = DecimalField(
        'Sender\'s New Balance',
        validators=[DataRequired(), NumberRange(min=0)],
        places=2
    )
    
    receiver_old_balance = DecimalField(
        'Receiver\'s Old Balance',
        validators=[DataRequired(), NumberRange(min=0)],
        places=2
    )
    
    receiver_new_balance = DecimalField(
        'Receiver\'s New Balance',
        validators=[DataRequired(), NumberRange(min=0)],
        places=2
    )
