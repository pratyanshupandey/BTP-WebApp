import Box from '@mui/material/Box';
import Grid from '@mui/material/Grid';
import Typography from '@mui/material/Typography'
import { Avatar } from '@mui/material';
import { Card, CardActions, CardContent, Button } from '@mui/material';
import { Container, Stack } from '@mui/material';

const AboutPage = () => {
  return (
    <Box sx={{ flexGrow: 1 }}>
      <Grid container>
        <Grid item xs={12} sm={12} md={12} lg={12} xl={12}>
          <Box sx={{
            flexGrow: 1,
            backgroundColor: 'primary.light',
            padding: 10
          }}>
            <Typography
              variant='h4'
              noWrap
              component="div">
              Detailed Project Description
            </Typography>
          </Box>
        </Grid>
        <Grid item xs={12} sm={12} md={12} lg={12} xl={12}>
          <Container>
            <Card sx={{ maxWidth: 275 }}>
              <CardContent>
                <Avatar
                  alt="Remy Sharp"
                  src="../assets/flight.jpg"
                  sx={{ width: 56, height: 56 }}
                />
                <Typography sx={{ fontSize: 14 }} color="text.secondary" gutterBottom>
                  Name
                </Typography>
                <Typography sx={{ fontSize: 14 }} color="text.secondary" gutterBottom>
                  Designation
                </Typography>
              </CardContent>
            </Card>

            <Card sx={{ maxWidth: 275 }}>
              <CardContent>
                <Avatar
                  alt="Remy Sharp"
                  src="../assets/flight.jpg"
                  sx={{ width: 56, height: 56 }}
                />
                <Typography sx={{ fontSize: 14 }} color="text.secondary" gutterBottom>
                  Name
                </Typography>
                <Typography sx={{ fontSize: 14 }} color="text.secondary" gutterBottom>
                  Designation
                </Typography>
              </CardContent>
            </Card>
          </Container>
        </Grid>
        <Grid item xs={12} sm={12} md={12} lg={12} xl={12}>
          <Box sx={{
            flexGrow: 1,
            backgroundColor: 'primary.light',
            padding: 10
          }}>
            <Typography
              variant='h4'
              noWrap
              component="div">
              Links
            </Typography>
          </Box>
        </Grid>
      </Grid>
    </Box>
  )
}

export default AboutPage